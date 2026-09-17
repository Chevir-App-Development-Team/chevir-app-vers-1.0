import fs from 'fs';
let html = fs.readFileSync('C:/Users/ELNUR/.gemini/antigravity/brain/f59a2294-a771-472b-9a89-91b3415ab9fb/.system_generated/steps/8/content.md', 'utf8');
const htmlStart = html.indexOf('<!DOCTYPE html>');
html = html.slice(htmlStart);
html = html.replace(/href="\/([^"]+)"/g, 'href="https://nsosyal.com/$1"');
html = html.replace(/src="\/([^"]+)"/g, 'src="https://nsosyal.com/$1"');
fs.writeFileSync('trending.html', html);
