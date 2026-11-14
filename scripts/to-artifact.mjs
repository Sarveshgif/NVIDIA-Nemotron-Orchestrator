// Converts the single-file Vite build into page content for an Artifact (no html/head/body wrapper).
import fs from 'fs';
const html = fs.readFileSync('dist-artifact/index.html', 'utf8');
const head = html.match(/<head>([\s\S]*?)<\/head>/i)[1];
const body = html.match(/<body>([\s\S]*?)<\/body>/i)[1];
const keep = head
  .replace(/<meta charset[^>]*>/i, '')
  .replace(/<meta name="viewport"[^>]*>/i, '');
const out = keep.trim() + '\n' + body.trim() + '\n';
fs.writeFileSync('dist-artifact/harvest-console.html', out);
console.log('wrote dist-artifact/harvest-console.html', (out.length / 1024).toFixed(0) + ' KB');
