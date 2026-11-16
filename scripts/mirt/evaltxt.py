import asyncio, subprocess, time
from playwright.async_api import async_playwright
srv = subprocess.Popen([".venv/bin/python","-m","http.server","8765","-d","ui","--bind","127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width":1500,"height":950})
        await pg.goto("http://127.0.0.1:8765/#reports"); await pg.wait_for_timeout(5000)
        t = await pg.locator("section.eval").inner_text(); i = t.find("Does more"); print(len(t), i); print(t[-1800:])
        el = pg.locator("section.eval h3").first; await el.scroll_into_view_if_needed(); await pg.wait_for_timeout(500)
        await pg.screenshot(path="shot_years.png")
        await b.close()
asyncio.run(main()); srv.terminate()
