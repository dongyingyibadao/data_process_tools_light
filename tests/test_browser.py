from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright

from test_workbench import make_episode


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_complete_browser_workflow_and_idle_network(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    cleaned = tmp_path / "cleaned"
    relative = "289/2026-08-30/task/episode_000001"
    make_episode(raw, relative, frames=120, fps=30)
    second_relative = "289/2026-08-30/task/episode_000002"
    make_episode(raw, second_relative, frames=24, fps=8)
    task = raw / "289/2026-08-30/task"
    for index in range(3, 130):
        (task / f"episode_{index:06d}" / "videos").mkdir(parents=True)

    port = free_port()
    environment = {**os.environ, "PORT": str(port), "HOST": "127.0.0.1"}
    server = subprocess.Popen(
        ["python", "-m", "uvicorn", "lightworkbench.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=Path(__file__).parents[1], env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise AssertionError("server did not start")

        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            api_requests: list[str] = []
            page.on("request", lambda request: api_requests.append(request.url) if "/api/" in request.url else None)
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_timeout(300)
            for _ in range(10):
                page.evaluate("followOperations()")
            page.locator("#queueToggle").click()
            page.locator("#queueClose").click()
            page.locator("#queueToggle").click()
            page.locator("#queueClose").click()
            event_requests = [url for url in api_requests if url.endswith("/api/operations/events")]
            assert len(event_requests) == 1
            assert not [url for url in api_requests if "/api/operations/" in url and url.endswith("/events") and not url.endswith("/operations/events")]

            transition = page.evaluate("""(() => {
                state.operationsInitialized = true;
                state.operations = new Map();
                state.notifiedTerminal.clear();
                const active = Array.from({length: 10}, (_, index) => ({
                    id: "test-" + index, status: "queued", progress: 0, message: "queued",
                    queuePosition: index + 1, episode: "task/episode_" + index,
                    mode: "no_trim", submittedAt: String(index),
                }));
                applyOperationsSnapshot({queued: active, running: [], completed: []});
                const completed = active.map((item) => ({...item, status: "completed", progress: 1,
                    result: {outputPath: "/tmp/" + item.id}}));
                applyOperationsSnapshot({queued: [], running: [], completed});
                const firstCount = state.notifiedTerminal.size;
                applyOperationsSnapshot({queued: [], running: [], completed});
                const secondCount = state.notifiedTerminal.size;
                applyOperationsSnapshot({queued: [], running: [], completed: []});
                return {firstCount, secondCount};
            })()""")
            assert transition == {"firstCount": 10, "secondCount": 10}

            page.locator("#sourceRoot").fill(str(raw))
            page.locator("#outputRoot").fill(str(cleaned))
            page.locator("#operator").fill("browser-tester")
            page.locator("#openRoot").click()
            page.locator(".folder-row", has_text="289").click()
            page.locator(".folder-row", has_text="2026-08-30").click()
            page.locator(".folder-row", has_text="task").click()
            expect(page.locator("#itemCount")).to_have_text("129")
            assert page.locator(".episode-row").count() <= 30
            page.get_by_role("button", name="episode_000001", exact=False).click()
            expect(page.locator("#validationBadge")).to_have_text("3 路同步有效")
            expect(page.locator("#video")).to_have_count(1)

            rapid_switch = page.evaluate("""async () => {
                const originalFetch = window.fetch;
                let aborted = false;
                window.fetch = (url, options = {}) => {
                    if (!String(url).includes("episode_000001")) return originalFetch(url, options);
                    return new Promise((resolve, reject) => {
                        const timer = setTimeout(() => originalFetch(url, options).then(resolve, reject), 500);
                        options.signal.addEventListener("abort", () => {
                            clearTimeout(timer);
                            aborted = true;
                            reject(new DOMException("Aborted", "AbortError"));
                        }, {once: true});
                    });
                };
                await Promise.all([
                    openEpisode("289/2026-08-30/task/episode_000001"),
                    openEpisode("289/2026-08-30/task/episode_000002"),
                ]);
                window.fetch = originalFetch;
                return {aborted, episode: state.detail && state.detail.episode};
            }""")
            assert rapid_switch == {"aborted": True, "episode": second_relative}
            expect(page.locator("#episodeName")).to_have_text("episode_000002")
            page.evaluate("openEpisode('289/2026-08-30/task/episode_000001')")
            expect(page.locator("#episodeName")).to_have_text("episode_000001")
            expect(page.locator("#validationBadge")).to_have_text("3 路同步有效")

            queue_regression = page.evaluate("""async ({queueOutput, mainOutput}) => {
                await api("/api/operations/settings", {
                    method: "PATCH", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({concurrency: 1}),
                });
                const validToken = state.detail.sourceToken;
                outputRoot.value = queueOutput;
                for (let index = 0; index < 10; index += 1) {
                    state.episode = index === 0
                        ? "289/2026-08-30/task/episode_000001"
                        : "289/2026-08-30/task/episode_" + String(index + 2).padStart(6, "0");
                    state.detail = {sourceToken: index === 0 ? validToken : "invalid-token"};
                    state.ranges = index === 0 ? [[0, 60]] : [];
                    await submit(index === 0 ? "trim" : "no_trim");
                }
                const started = performance.now();
                await openEpisode("289/2026-08-30/task/episode_000002");
                const elapsed = performance.now() - started;
                outputRoot.value = mainOutput;
                return {elapsed, episode: state.detail && state.detail.episode};
            }""", {"queueOutput": str(cleaned / "queue-regression"), "mainOutput": str(cleaned)})
            assert queue_regression["elapsed"] < 2_000
            assert queue_regression["episode"] == second_relative
            event_requests = [url for url in api_requests if url.endswith("/api/operations/events")]
            assert len(event_requests) == 1
            assert not [url for url in api_requests if "/api/operations/" in url and url.endswith("/events") and not url.endswith("/operations/events")]
            expect(page.locator("#validationBadge")).to_have_text("3 路同步有效")
            expect(page.locator("#video")).to_have_attribute("src", re.compile("episode_000002"))

            page.evaluate("openEpisode('289/2026-08-30/task/episode_000001')")
            expect(page.locator("#validationBadge")).to_have_text("3 路同步有效")
            page.get_by_role("button", name="rgbd_head_color", exact=True).click()
            expect(page.locator("button.stream-tab.active")).to_have_text("rgbd_head_color")
            track = page.locator("#rangeTrack")
            track.scroll_into_view_if_needed()
            box = track.bounding_box()
            hidden_displays = page.evaluate("""[
                getComputedStyle(document.querySelector('#browserEmpty')).display,
                getComputedStyle(document.querySelector('#welcome')).display,
                getComputedStyle(document.querySelector('#videoUnavailable')).display,
            ]""")
            assert hidden_displays == ["none", "none", "none"]
            desktop = page.evaluate("""({editorClient: editor.clientHeight, editorScroll: editor.scrollHeight,
                buttonBottom: trim.getBoundingClientRect().bottom, viewport: innerHeight})""")
            page.screenshot(path=str(tmp_path / "desktop-1366x768.png"))
            assert desktop["editorScroll"] <= desktop["editorClient"]
            assert desktop["buttonBottom"] <= desktop["viewport"]
            assert box is not None
            page.mouse.move(box["x"] + box["width"] * .12, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * .38, box["y"] + box["height"] / 2)
            page.mouse.up()
            preview_jumps = page.evaluate("""(() => {
                state.ranges = normalizeRanges([[1, 30], [50, 70]]);
                renderRanges();
                return [0, 1, 29, 30, 49, 50, 69, 70].map(keptFrameAtOrAfter);
            })()""")
            assert preview_jumps == [0, 30, 30, 30, 49, 70, 70, 70]
            expect(page.locator(".range-chip")).to_have_count(2)
            page.locator("#preview").click()
            expect(page.locator("#preview")).to_have_text("退出预览")
            expect(page.locator("#video")).not_to_have_js_property("paused", True)

            api_requests.clear()
            page.wait_for_timeout(30_000)
            background = [url for url in api_requests if "/media/" not in url]
            assert background == []

            page.locator("#trim").click()
            expect(page.locator("#operationStatus")).to_have_text("completed", timeout=30_000)
            expect(page.locator("#episodeName")).to_have_text("episode_000001")
            expect(page.locator(".range-chip")).to_have_count(2)

            page.locator("#noTrim").click()
            expect(page.locator("#overwriteDialog")).to_be_visible()
            page.locator("#confirmOverwrite").click()
            expect(page.locator("#operationStatus")).to_have_text("running", timeout=10_000)
            expect(page.locator("#operationStatus")).to_have_text("completed", timeout=30_000)

            page.set_viewport_size({"width": 1920, "height": 1080})
            page.wait_for_timeout(200)
            wide = page.evaluate("""({editorClient: editor.clientHeight, editorScroll: editor.scrollHeight,
                buttonBottom: trim.getBoundingClientRect().bottom, viewport: innerHeight})""")
            assert wide["editorScroll"] <= wide["editorClient"]
            assert wide["buttonBottom"] <= wide["viewport"]
            page.screenshot(path=str(tmp_path / "desktop-1920x1080.png"))

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(200)
            dimensions = page.evaluate("({scroll: document.documentElement.scrollWidth, width: innerWidth})")
            assert dimensions["scroll"] <= dimensions["width"]
            page.screenshot(path=str(tmp_path / "mobile.png"), full_page=True)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
