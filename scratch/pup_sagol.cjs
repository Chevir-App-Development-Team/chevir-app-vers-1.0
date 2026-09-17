const puppeteer = require('puppeteer');

(async () => {
    try {
        const browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox'] 
        });
        const page = await browser.newPage();
        
        console.log('Navigating to sagol...');
        await page.goto('https://www.jestdili.az/az/lugat/word/sagol', { waitUntil: 'networkidle2' });
        
        const vids = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('video')).map(v => v.src);
        });
        console.log('Videos on sagol page:', vids);
        
        await browser.close();
    } catch (e) {
        console.error(e);
    }
})();
