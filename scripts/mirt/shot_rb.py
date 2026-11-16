import asyncio, subprocess, time
from playwright.async_api import async_playwright
srv = subprocess.Popen([".venv/bin/python","-m","http.server","8766","-d","rbui","--bind","127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width":1600,"height":1000})
        errs=[]; pg.on("console", lambda m: errs.append(m.type+": "+m.text) if m.type in ("error","warning") else None); pg.on("pageerror", lambda e: errs.append("PAGEERROR "+str(e)))
        reqs=[]; pg.on("request", lambda r: reqs.append(r.url) if "barc/" in r.url else None)
        await pg.goto("http://127.0.0.1:8766/#rulebook", wait_until="networkidle"); await pg.wait_for_timeout(3500)
        print("label file requested on predictions tab:", any("eval" in u for u in reqs))
        print("rows:", await pg.locator(".rb-table tbody tr").count(), "map polygons:", await pg.locator(".rb-map path.leaflet-interactive").count())
        await pg.screenshot(path="shot_rb_list.png")
        await pg.locator(".rb-table tbody tr", has_text="LIKELY GRAIN").first.click(); await pg.wait_for_timeout(1500)
        await pg.screenshot(path="shot_rb_detail.png", full_page=True)
        print("detail charts:", await pg.locator(".rb-chart .recharts-surface").count(), "checks:", await pg.locator(".rb-checks details").count())
        await pg.get_by_role("tab", name="Evaluation (reference labels)").click(); await pg.wait_for_timeout(2000)
        print("eval requested after tab:", any("eval" in u for u in reqs))
        await pg.screenshot(path="shot_rb_eval.png", full_page=True)
        print("\n".join(errs[:15]) or "no console errors")
        await b.close()
asyncio.run(main()); srv.terminate()
