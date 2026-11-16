import asyncio, subprocess, time, statistics as st
from playwright.async_api import async_playwright
srv = subprocess.Popen([".venv/bin/python","-m","http.server","8765","-d","ui","--bind","127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width":1500,"height":950})
        t0=time.perf_counter(); await pg.goto("http://127.0.0.1:8765/"); await pg.wait_for_selector("text=Analysis ready", timeout=60000); load=time.perf_counter()-t0
        ids = await pg.evaluate("() => [...document.querySelectorAll(\".q-row, [data-field-id]\")].length")
        # time field switching via the drawer Next button (cached evidence)
        await pg.locator(".hint-pick").first.click(); await pg.wait_for_selector(".drawer, [aria-label*=Field]", timeout=10000)
        times=[]
        for _ in range(15):
            cur = await pg.locator("h2.mono, .drawer-title, .mono").first.inner_text()
            t=time.perf_counter(); await pg.get_by_role("button", name="Next field").click()
            await pg.wait_for_function("(c) => { const e=document.querySelector(\"h2.mono, .drawer-title, .mono\"); return e && e.innerText!==c }", arg=cur, timeout=5000)
            times.append((time.perf_counter()-t)*1000)
        print(f"initial load to analysis ready: {load:.2f} s (9 MB JSON, localhost, headless Chromium, lab workstation)")
        print(f"field switch (cached evidence): median {st.median(times):.0f} ms, max {max(times):.0f} ms over {len(times)}")
        await b.close()
asyncio.run(main()); srv.terminate()
