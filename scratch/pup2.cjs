const puppeteer = require('puppeteer');

(async () => {
    try {
        const browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox'] 
        });
        const page = await browser.newPage();
        
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
        
        const vids = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('video')).map(v => v.src);
        });
        console.log('Videos on page:', vids);
        
        await browser.close();
    } catch (e) {
        console.error(e);
    }
})();
