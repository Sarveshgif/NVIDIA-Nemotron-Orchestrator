import asyncio, subprocess, time
from playwright.async_api import async_playwright
srv = subprocess.Popen([".venv/bin/python","-m","http.server","8765","-d","ui","--bind","127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width":1500,"height":950}); errs=[]
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://127.0.0.1:8765/"); await pg.wait_for_selector("text=Analysis ready", timeout=30000)
        await pg.select_option("#sel-region", "tulare-ca"); await pg.wait_for_selector("text=Analysis ready", timeout=30000); await pg.wait_for_timeout(2500)
        txt = await pg.locator(".acreage-stats").inner_text(); print("TULARE:", " | ".join(txt.split(chr(10))[:9]))
        print("year:", await pg.locator("#sel-year").input_value(), "badge:", await pg.locator(".demo-badge").inner_text())
        await pg.screenshot(path="shot_ca.png"); print("errors:", errs or "none"); await b.close()
asyncio.run(main()); srv.terminate()
