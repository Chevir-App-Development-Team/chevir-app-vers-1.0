import fs from 'fs';
let content = fs.readFileSync('C:/Users/ELNUR/.gemini/antigravity/brain/f59a2294-a771-472b-9a89-91b3415ab9fb/.system_generated/steps/8/content.md', 'utf8');
const htmlStart = content.indexOf('<!DOCTYPE html>');
let html = content.slice(htmlStart);
html = html.replace(/href="\//g, 'href="https://nsosyal.com/');
html = html.replace(/src="\//g, 'src="https://nsosyal.com/');

// Remove all script tags so Next.js doesn't clear the DOM!
html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

fs.writeFileSync('trending/index.html', html);
