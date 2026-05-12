"""
seif-online.com Web Scraper
============================
Scrapes two datasets:
  1. products_data.csv  — product name, price, SKU, description, category, subcategory, URL
  2. images/            — all product images saved locally + images_index.csv with metadata

Requirements:
    pip install playwright pandas requests tqdm
    playwright install chromium

Usage:
    python seif_scraper.py
"""

import asyncio
import json
import os
import re
import time
import csv
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL        = "https://seif-online.com/en"
OUTPUT_DIR      = Path("seif_output")
IMAGES_DIR      = OUTPUT_DIR / "images"
PRODUCTS_CSV    = OUTPUT_DIR / "products_data.csv"
IMAGES_CSV      = OUTPUT_DIR / "images_index.csv"
HEADLESS        = True          # Set False to watch the browser
PAGE_TIMEOUT    = 30_000        # ms
NAV_PAUSE       = 1.5           # seconds between page navigations
MAX_RETRIES     = 3
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:80]


def download_image(url: str, dest: Path, session: requests.Session) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists():
        return True
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  ✗ Failed to download {url}: {e}")
                return False
            time.sleep(1)
    return False


async def safe_goto(page, url: str) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            await page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT)
            return True
        except PlaywrightTimeoutError:
            print(f"  Timeout on {url}, retry {attempt+1}/{MAX_RETRIES}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"  Error on {url}: {e}")
            return False
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Discover categories & subcategories
# ═══════════════════════════════════════════════════════════════════════════

async def get_categories(page) -> list[dict]:
    """
    Returns a list of dicts:
        {
          'category': str,
          'subcategory': str | None,
          'url': str
        }
    These are the leaf-level browsable URLs we will paginate through.
    """
    print("\n[1/3] Discovering categories…")
    await safe_goto(page, BASE_URL)
    await asyncio.sleep(NAV_PAUSE)

    # Try to find all category/subcategory links in the nav / sidebar
    # seif-online.com uses a typical e-commerce nav with dropdowns
    nav_links: list[dict] = []

    # Grab all nav anchor tags
    anchors = await page.query_selector_all("a[href]")
    seen = set()

    for a in anchors:
        href = await a.get_attribute("href") or ""
        text = (await a.inner_text()).strip()

        # Filter for category-style links  /en/categories/... or /en/c/...
        if not href:
            continue
        full = href if href.startswith("http") else urljoin(BASE_URL, href)
        if "seif-online.com" not in full:
            continue
        if any(seg in full for seg in ["/categories/", "/category/", "/c/", "/shop/", "/collection"]):
            if full not in seen and text:
                seen.add(full)
                nav_links.append({"raw_url": full, "link_text": text})

    print(f"   Found {len(nav_links)} raw category links from nav.")

    # Deduplicate & build hierarchy by URL depth
    categories = []
    for item in nav_links:
        url   = item["raw_url"].rstrip("/")
        parts = urlparse(url).path.strip("/").split("/")
        # parts example: ['en', 'categories', 'electronics', 'phones']
        # strip lang prefix
        parts = [p for p in parts if p not in ("en", "ar")]

        if len(parts) >= 3:
            category    = parts[1].replace("-", " ").title()
            subcategory = parts[2].replace("-", " ").title() if len(parts) >= 4 else None
        elif len(parts) == 2:
            category    = parts[1].replace("-", " ").title()
            subcategory = None
        else:
            continue

        categories.append({
            "category":    category,
            "subcategory": subcategory,
            "url":         item["raw_url"],
        })

    # If we found nothing through nav, fall back to the /en/categories/ listing page
    if not categories:
        print("   Nav approach yielded nothing — trying /en/categories/ page…")
        categories = await get_categories_from_listing_page(page)

    print(f"   → {len(categories)} category/subcategory URLs to scrape.")
    return categories


async def get_categories_from_listing_page(page) -> list[dict]:
    """Fallback: parse the categories listing page."""
    await safe_goto(page, "https://seif-online.com/en/categories/")
    await asyncio.sleep(NAV_PAUSE)

    cards = await page.query_selector_all("a[href]")
    results = []
    seen = set()
    for card in cards:
        href = await card.get_attribute("href") or ""
        text = (await card.inner_text()).strip()
        if not href or not text:
            continue
        full = href if href.startswith("http") else urljoin(BASE_URL, href)
        if full in seen:
            continue
        seen.add(full)
        results.append({
            "category":    text,
            "subcategory": None,
            "url":         full,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Scrape products from a single category page (with pagination)
# ═══════════════════════════════════════════════════════════════════════════

async def scrape_products_from_category(page, cat: dict) -> list[dict]:
    """
    Paginates through a category URL and collects all product detail URLs.
    Returns list of partial product dicts (url, category, subcategory).
    """
    product_entries = []
    url = cat["url"]
    page_num = 1

    while url:
        ok = await safe_goto(page, url)
        if not ok:
            break
        await asyncio.sleep(NAV_PAUSE)

        # Collect product card links — typically anchor tags inside .product-card / .item etc.
        # We cast a wide net and filter by URL pattern
        anchors = await page.query_selector_all("a[href]")
        found_on_page = 0
        seen_on_page = set()

        for a in anchors:
            href = await a.get_attribute("href") or ""
            full = href if href.startswith("http") else urljoin(BASE_URL, href)
            if "seif-online.com" not in full:
                continue
            # Product URLs usually contain /products/ or /p/ or /item/
            if any(seg in full for seg in ["/products/", "/product/", "/p/", "/item/"]):
                if full not in seen_on_page:
                    seen_on_page.add(full)
                    found_on_page += 1
                    product_entries.append({
                        "product_url": full,
                        "category":    cat["category"],
                        "subcategory": cat["subcategory"],
                    })

        print(f"     Page {page_num}: {found_on_page} products")

        # Find next-page link
        next_btn = await page.query_selector(
            "a[aria-label='Next'], a.next, a[rel='next'], "
            "button.next, [class*='pagination'] a:last-child"
        )
        if next_btn:
            next_href = await next_btn.get_attribute("href")
            if next_href and next_href != url:
                url = next_href if next_href.startswith("http") else urljoin(BASE_URL, next_href)
                page_num += 1
                continue
        break  # no next page

    return product_entries


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Scrape a single product page
# ═══════════════════════════════════════════════════════════════════════════

async def scrape_product_detail(page, entry: dict) -> dict | None:
    """
    Visits a product page and extracts all fields.
    Returns a dict or None on failure.
    """
    ok = await safe_goto(page, entry["product_url"])
    if not ok:
        return None
    await asyncio.sleep(0.8)

    async def text(selector: str) -> str:
        el = await page.query_selector(selector)
        return (await el.inner_text()).strip() if el else ""

    async def attr(selector: str, attribute: str) -> str:
        el = await page.query_selector(selector)
        return (await el.get_attribute(attribute) or "").strip() if el else ""

    # ── Name ────────────────────────────────────────────────────────────────
    name = await text("h1") or await text("[class*='product-name']") or await text("[class*='title']")

    # ── Price ───────────────────────────────────────────────────────────────
    price = (
        await text("[class*='price']:not([class*='old']):not([class*='was'])")
        or await text("[class*='amount']")
        or await text("[itemprop='price']")
    )

    # ── SKU / Barcode ────────────────────────────────────────────────────────
    sku = await text("[class*='sku']") or await text("[class*='barcode']") or await attr("[itemprop='sku']", "content")

    # ── Description ─────────────────────────────────────────────────────────
    description = (
        await text("[class*='description']")
        or await text("[itemprop='description']")
        or await text("[class*='details']")
    )

    # ── Brand ────────────────────────────────────────────────────────────────
    brand = await text("[class*='brand']") or await attr("[itemprop='brand']", "content")

    # ── Rating ───────────────────────────────────────────────────────────────
    rating = await attr("[itemprop='ratingValue']", "content") or await text("[class*='rating']")

    # ── Images ───────────────────────────────────────────────────────────────
    img_elements = await page.query_selector_all(
        "[class*='product'] img, [class*='gallery'] img, [class*='slider'] img, "
        "[id*='product'] img, main img"
    )
    image_urls = []
    for img in img_elements:
        src = (
            await img.get_attribute("src")
            or await img.get_attribute("data-src")
            or await img.get_attribute("data-lazy")
            or ""
        )
        if src and src.startswith("http") and src not in image_urls:
            # Filter out tiny icons / logos
            if not any(skip in src.lower() for skip in ["logo", "icon", "placeholder", "spinner", "loading"]):
                image_urls.append(src)

    # ── Availability ─────────────────────────────────────────────────────────
    availability = (
        await text("[class*='stock']")
        or await text("[class*='availab']")
        or await attr("[itemprop='availability']", "content")
    )

    return {
        "name":          name,
        "price":         price,
        "sku":           sku,
        "brand":         brand,
        "category":      entry["category"],
        "subcategory":   entry["subcategory"],
        "description":   description[:500].replace("\n", " ") if description else "",
        "rating":        rating,
        "availability":  availability,
        "image_urls":    "|".join(image_urls),  # pipe-separated in CSV
        "product_url":   entry["product_url"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Download all images
# ═══════════════════════════════════════════════════════════════════════════

def download_all_images(products: list[dict]) -> list[dict]:
    """Download product images and return image metadata rows."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SeifScraper/1.0)"})

    image_records = []
    print(f"\n[3/3] Downloading images…")

    for product in tqdm(products, desc="Products"):
        if not product.get("image_urls"):
            continue

        urls = [u for u in product["image_urls"].split("|") if u]
        product_slug = slugify(product.get("name") or "product")

        for idx, img_url in enumerate(urls):
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            filename = f"{product_slug}_{idx+1}{ext}"
            dest = IMAGES_DIR / filename

            success = download_image(img_url, dest, session)
            image_records.append({
                "product_name": product.get("name", ""),
                "category":     product.get("category", ""),
                "subcategory":  product.get("subcategory", ""),
                "image_index":  idx + 1,
                "image_url":    img_url,
                "local_file":   str(dest) if success else "",
                "downloaded":   success,
                "product_url":  product.get("product_url", ""),
            })

    return image_records


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  seif-online.com Scraper")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        # ── 1. Get all categories ──────────────────────────────────────────
        categories = await get_categories(page)
        if not categories:
            print("ERROR: No categories found. Check the site structure manually.")
            await browser.close()
            return

        # ── 2. For each category, collect product URLs ─────────────────────
        print("\n[2/3] Collecting product links from each category…")
        all_product_entries: list[dict] = []

        for cat in tqdm(categories, desc="Categories"):
            entries = await scrape_products_from_category(page, cat)
            all_product_entries.extend(entries)

        # Deduplicate by URL (a product may appear in multiple categories)
        seen_urls = {}
        deduped = []
        for e in all_product_entries:
            if e["product_url"] not in seen_urls:
                seen_urls[e["product_url"]] = True
                deduped.append(e)

        print(f"\n   → {len(deduped)} unique product pages found.")

        # ── 3. Scrape each product detail page ────────────────────────────
        products: list[dict] = []
        failed = 0

        for entry in tqdm(deduped, desc="Scraping products"):
            product = await scrape_product_detail(page, entry)
            if product:
                products.append(product)
            else:
                failed += 1

        await browser.close()

    print(f"\n   Scraped {len(products)} products ({failed} failed).")

    # ── 4. Save products CSV ───────────────────────────────────────────────
    if products:
        df_products = pd.DataFrame(products)
        df_products.to_csv(PRODUCTS_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✔  Products dataset saved → {PRODUCTS_CSV}")
        print(df_products[["name", "price", "category", "subcategory"]].head(10).to_string(index=False))

    # ── 5. Download images & save image index ─────────────────────────────
    if products:
        image_records = download_all_images(products)
        df_images = pd.DataFrame(image_records)
        df_images.to_csv(IMAGES_CSV, index=False, encoding="utf-8-sig")
        total_downloaded = df_images["downloaded"].sum()
        print(f"\n✔  Images index saved   → {IMAGES_CSV}")
        print(f"   Downloaded {total_downloaded}/{len(image_records)} images → {IMAGES_DIR}/")

    print("\n Done!")
    print(f"   Output folder: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())