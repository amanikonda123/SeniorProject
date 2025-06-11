import os
import time
from apify_client import ApifyClient

class WalmartScraper:
    """
    Scrapes Walmart customer reviews using Apify's Walmart Reviews Scraper Actor.

    By default, this uses Apify's public actor "apify/walmart-reviews-scraper"—no
    additional setup is required if you have an Apify account and valid APIFY_TOKEN.

    If you prefer to customize the actor logic or avoid rate limits, you can fork the
    public actor in your Apify account and override `APIFY_ACTOR_ID` (or pass `actor_id`
    to the constructor) with your own actor handle.

    Prerequisites:
      - An Apify account (free tier available).
      - APIFY_TOKEN environment variable set to your Apify API token.
      - (Optional) APIFY_ACTOR_ID environment variable set to your custom actor ID.
    """
    def __init__(self, apify_token: str = None, actor_id: str = None):
        # Use provided token or fall back to environment
        self.token = apify_token or os.getenv("APIFY_TOKEN")
        if not self.token:
            raise ValueError("APIFY_TOKEN environment variable must be set.")
        # Use provided actor_id, environment variable, or default to public actor
        self.actor_id = actor_id or os.getenv("APIFY_ACTOR_ID")
        self.client = ApifyClient(self.token)
 
    def get_reviews(self, product_sku: str) -> list:
        """
        Runs the Apify Actor and retrieves all reviews, polling until completion.

        Args:
            product_sku: 8-digit Walmart SKU number (e.g. "17235783").
            max_reviews: Optional cap on number of reviews.

        Returns:
            A list of review dictionaries as returned by the actor's dataset.
        """
        run_input = {
            "startUrls": [
                {"url": f"https://www.walmart.com/search?q={product_sku}", "method": "GET"}
            ],
            "reviewsSortType": "relevancy"
        }
        # Launch actor run
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        run_id = run.get("id")
        if not run_id:
            raise RuntimeError(
                "Failed to launch Apify actor run; check actor_id and APIFY_TOKEN."
            )
        # Poll run status
        while True:
            info = self.client.run(run_id).get()
            status = info.get("status")
            if status == "SUCCEEDED":
                break
            if status in ("FAILED", "TIMED_OUT", "ABORTED"):  
                raise RuntimeError(f"Actor run failed with status: {status}")
            time.sleep(5)
        # Retrieve dataset
        dataset_id = info.get("defaultDatasetId")
        if not dataset_id:
            raise RuntimeError(
                "Actor run did not produce a dataset; check Actor configuration."
            )
        
         # Page through dataset items
        reviews = []
        offset = 0
        batch_size = 1000
        while True:
            list_page = self.client.dataset(dataset_id).list_items(limit=batch_size, offset=offset)
            batch = list_page.items
            if not batch:
                break
            reviews.extend(batch)
            offset += len(batch)

        return reviews

