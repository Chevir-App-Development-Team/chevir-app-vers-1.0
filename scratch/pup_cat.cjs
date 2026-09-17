const puppeteer = require('puppeteer');

(async () => {
    try {
        const browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox'] 
        });
        const page = await browser.newPage();
        
        console.log('Navigating to category...');
        await page.goto('https://www.jestdili.az/az/lugat/10000000-0000-0000-0000-000000000024', { waitUntil: 'networkidle2' });
        
        const words = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('a')).map(a => {
                return { text: a.innerText, href: a.href };
            }).filter(x => x.href.includes('/word/'));
        });
        console.log('Words:', words);
        
        await browser.close();
    } catch (e) {
        console.error(e);
    }
})();
