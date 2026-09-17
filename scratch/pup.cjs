const puppeteer = require('puppeteer');

(async () => {
    try {
        const browser = await puppeteer.launch({ headless: true });
        const page = await browser.newPage();
        
        // Listen to all network requests
        page.on('response', async (response) => {
            const url = response.url();
            if (url.includes('api') || url.includes('words') || url.includes('salam')) {
                console.log('Network response:', url);
                if (url.includes('.mp4')) {
                    console.log('FOUND MP4:', url);
                }
            }
        });
        
        console.log('Navigating to salam...');
        await page.goto('https://www.jestdili.az/az/lugat/word/salam', { waitUntil: 'networkidle2' });
        
        // Extract all videos
        const vids = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('video')).map(v => v.src);
        });
        console.log('Videos on page:', vids);
        
        // Look for any links ending in mp4
        const links = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('a')).map(a => a.href).filter(h => h.includes('.mp4'));
        });
        console.log('MP4 links on page:', links);
        
        // Get the entire HTML body just to check if "salam" is there
        const bodyText = await page.evaluate(() => document.body.innerText);
        if (bodyText.includes('salam') || bodyText.includes('Salam')) {
            console.log('Found "salam" in page text!');
        } else {
            console.log('"salam" NOT FOUND in page text!');
        }
        
        await browser.close();
    } catch (e) {
        console.error(e);
    }
})();
