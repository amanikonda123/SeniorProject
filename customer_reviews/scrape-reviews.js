// scrape-reviews.js
const cheerio = require('cheerio');
const fs      = require('fs');

function scrapeReviews(htmlFile = 'page.html', outJson = 'reviews.json') {
  const html = fs.readFileSync(htmlFile, 'utf-8');
  const $    = cheerio.load(html);
  const reviews = [];

  $('#item-review-section li').each((_, li) => {
    const rating     = $(li).find('.rating-stars').text().trim();
    const reviewText = $(li).find('.review-text').text().trim();
    if (rating && reviewText) {
      reviews.push({ rating, review: reviewText });
    }
  });

  fs.writeFileSync(outJson, JSON.stringify(reviews, null, 2));
  console.log(`Extracted ${reviews.length} reviews to ${outJson}`);
}

// Run it:
scrapeReviews();
