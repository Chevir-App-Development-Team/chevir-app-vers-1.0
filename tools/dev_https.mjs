// Telefonla sınaq: kamera yalnız təhlükəsiz ünvanda (https və ya localhost) açılır. Bu skript öz-özünə imzalanmış
// sertifikat yaradır (.cert/, git-ə düşmür; bu kompüterin şəbəkə ünvanları üçün) və Vite-i https ilə, şəbəkəyə açıq
// işə salır. Telefon eyni Wi-Fi-da: https://<kompüterin ünvanı>:5174/danis/?debug=1, xəbərdarlıqda "davam et".
//   npm run dev:https [-- --port 5174]
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { networkInterfaces } from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const root = new URL('../', import.meta.url);
const dir = new URL('.cert/', root);
const keyFile = new URL('key.pem', dir), certFile = new URL('cert.pem', dir), sanFile = new URL('san.txt', dir);
const args = process.argv.slice(2);
const port = Number(args.includes('--port') ? args[args.indexOf('--port') + 1] : 5174);

const ips = Object.values(networkInterfaces()).flat().filter((a) => a && a.family === 'IPv4').map((a) => a.address);
const san = ['DNS:localhost', ...ips.map((ip) => `IP:${ip}`)].join(',');

if (!existsSync(certFile) || !existsSync(sanFile) || readFileSync(sanFile, 'utf8') !== san) {
  mkdirSync(dir, { recursive: true });
  // openssl: PATH-də, yoxsa Git for Windows-un içindəki
  const candidates = ['openssl', 'C:/Program Files/Git/mingw64/bin/openssl.exe', 'C:/Program Files/Git/usr/bin/openssl.exe'];
  const opensslArgs = ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '365', '-subj', '/CN=chevir-dev',
    '-addext', `subjectAltName=${san}`, '-keyout', fileURLToPath(keyFile), '-out', fileURLToPath(certFile)];
  const ok = candidates.some((bin) => spawnSync(bin, opensslArgs, { stdio: 'ignore' }).status === 0);
  if (!ok) {
    console.error('openssl tapılmadı: Git for Windows-u quraşdırın və ya openssl-i PATH-ə əlavə edin');
    process.exit(1);
  }
  writeFileSync(sanFile, san);
  console.log(`Sertifikat yaradıldı (.cert/): ${san}`);
}

const server = await createServer({
  server: { host: true, port, strictPort: true, https: { key: readFileSync(keyFile), cert: readFileSync(certFile) } },
});
await server.listen();
server.printUrls();
console.log('Telefonda: yuxarıdakı Network ünvanı + /danis/?debug=1 (sertifikat xəbərdarlığında "davam et")');
