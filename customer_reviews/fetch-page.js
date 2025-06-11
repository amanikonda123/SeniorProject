// fetch-page.js
const axios = require('axios');
const fs   = require('fs');

async function fetchPage(url, outFile = 'page.html') {
  try {
    const { data: html } = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible)' }
    });
    fs.writeFileSync(outFile, html, 'utf-8');
    console.log(`Saved HTML to ${outFile}`);
  } catch (err) {
    console.error('Error fetching page:', err.message);
  }
}

// Example usage:
const productUrl = process.argv[2];
if (!productUrl) {
  console.error('Usage: node fetch-page.js <Walmart product URL>');
  process.exit(1);
}
fetchPage(productUrl);
