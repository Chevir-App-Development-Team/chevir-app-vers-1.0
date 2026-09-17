from playwright.sync_api import sync_playwright
import time

def fetch_videos():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        videos_found = {}
        
        # Intercept network responses to find video or api responses
        def handle_response(response):
            url = response.url
            if '/api/v1/' in url or '/api/' in url:
                try:
                    if 'json' in response.headers.get('content-type', ''):
                        data = response.json()
                        print(f"API Response from {url}:", str(data)[:200])
                except:
                    pass
        
        page.on("response", handle_response)
        
        print("Navigating to Salam page...")
        page.goto("https://www.jestdili.az/az/lugat/word/salam", wait_until="networkidle")
        time.sleep(2)
        
        # Also let's try getting the src of the video element
        video_elements = page.locator("video")
        for i in range(video_elements.count()):
            src = video_elements.nth(i).get_attribute("src")
            print("Found video src:", src)
            
        print("Navigating to Sağol page...")
        page.goto("https://www.jestdili.az/az/lugat/word/sa%C4%9Fol", wait_until="networkidle")
        time.sleep(2)
        
        video_elements = page.locator("video")
        for i in range(video_elements.count()):
            src = video_elements.nth(i).get_attribute("src")
            print("Found video src for sağol:", src)
            
        browser.close()

if __name__ == "__main__":
    fetch_videos()
