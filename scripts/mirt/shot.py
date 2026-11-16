import asyncio, subprocess, time, sys
from playwright.async_api import async_playwright
srv = subprocess.Popen([".venv/bin/python","-m","http.server","8765","-d","ui","--bind","127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width":1500,"height":950})
        errs=[]; pg.on("console", lambda m: errs.append(m.type+": "+m.text) if m.type in ("error","warning") else None); pg.on("pageerror", lambda e: errs.append("PAGEERROR "+str(e)))
        await pg.goto("http://127.0.0.1:8765/", wait_until="networkidle"); await pg.wait_for_timeout(4000)
        await pg.screenshot(path="shot_workspace.png")
        # open first showcase field from the hint
        btn = pg.locator(".hint-pick").first
        if await btn.count(): await btn.click(); await pg.wait_for_timeout(2500); await pg.screenshot(path="shot_field.png", full_page=False)
        await pg.goto("http://127.0.0.1:8765/#reports"); await pg.wait_for_timeout(4000)
        await pg.screenshot(path="shot_reports.png", full_page=True)
        await pg.goto("http://127.0.0.1:8765/#labels"); await pg.wait_for_timeout(3000)
        await pg.screenshot(path="shot_labels.png")
        print("\n".join(errs[:15]) or "no console errors")
        await b.close()
asyncio.run(main()); srv.terminate()
