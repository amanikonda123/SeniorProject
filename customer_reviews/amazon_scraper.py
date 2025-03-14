import os
import argparse
from selenium import webdriver
from amazoncaptcha import AmazonCaptcha
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service  # NEW IMPORT
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import streamlit as st

class AmazonScraper:
    __wait_time = 0.5

    __amazon_search_url = 'https://www.amazon.com/s?k='
    __amazon_review_url = 'https://www.amazon.com/product-reviews/'

    __star_page_suffix = {
        5: '/ref=cm_cr_unknown?filterByStar=five_star&pageNumber=',
        4: '/ref=cm_cr_unknown?filterByStar=four_star&pageNumber=',
        3: '/ref=cm_cr_unknown?filterByStar=three_star&pageNumber=',
        2: '/ref=cm_cr_unknown?filterByStar=two_star&pageNumber=',
        1: '/ref=cm_cr_unknown?filterByStar=one_star&pageNumber=',
    }

    def __init__(self):
        pass

    def __get_amazon_search_page(self, search_query: str, headless: bool = True):
        # setting up a headless web driver to get search query
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=options)

        url = AmazonScraper.__amazon_search_url + '+'.join(search_query.split())
        driver.get(url)
        driver.implicitly_wait(AmazonScraper.__wait_time)

        html_page = driver.page_source
        driver.quit()

        return html_page

    def __get_closest_product_asin(self, html_page: str):
        soup = BeautifulSoup(html_page, 'lxml')

        # data-asin grabs products, while data-avar filters out sponsored ads
        listings = soup.findAll('div', attrs={'data-asin': True, 'data-avar': False})
        asin_values = [single_listing['data-asin'] for single_listing in listings if single_listing['data-asin']]

        assert len(asin_values) > 0
        return asin_values[0]

    def __get_rated_reviews(self, url: str, headless: bool = True):
        # 1. Create ChromeOptions
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        
        # 2. Point to your Chrome installation
        #    (Adjust the path if your Chrome is installed elsewhere)
        options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

        # 4. Pass both service and options
        driver = webdriver.Chrome(options=options)

        driver.get("https://www.amazon.com/errors/validateCaptcha")

        # Get the captcha image link
        link = driver.find_element(By.XPATH, '//div[@class="a-row a-text-center"]/img').get_attribute("src")

        # Solve the captcha using amazon captcha
        captcha = AmazonCaptcha.fromlink(link)
        captcha_value = AmazonCaptcha.solve(captcha)

        # Enter the solved captcha text
        input_field = driver.find_element(By.ID, "captchacharacters")
        input_field.send_keys(captcha_value)

        # Click the submit button
        button = driver.find_element(By.CLASS_NAME, "a-button-text")
        button.click()

        driver.get("https://www.amazon.com")

        wait = WebDriverWait(driver, 10)  # up to 10 seconds
        account_lists_button = wait.until(
            EC.element_to_be_clickable((By.ID, "nav-link-accountList"))
        )
        account_lists_button.click()

        # 3. Wait for the email field on the sign-in page, then enter email
        email_field = wait.until(EC.presence_of_element_located((By.ID, "ap_email")))
        email_field.send_keys(os.environ.get("EMAIL"))

        continue_button = driver.find_element(By.ID, "continue")
        continue_button.click()

        # 4. Enter password
        password_field = wait.until(EC.presence_of_element_located((By.ID, "ap_password")))
        password_field.send_keys(os.environ.get("PASSWORD"))

        login_button = driver.find_element(By.ID, "signInSubmit")
        login_button.click()

        driver.get(url)
        wait = WebDriverWait(driver, 30)  # up to 15 seconds
        wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-hook="review"]'))
        )

        st.write("Final URL before scraping:", driver.current_url)
        html_page = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html_page, 'lxml')
        html_reviews = soup.findAll('div', attrs={"data-hook": "review"})

        st.write("HTML Reviews (Before):", html_reviews)
        reviews = []
        # extract text from various span tags and clean up newlines in their strings
        for html_review in html_reviews:
            name = html_review.find('span', class_='a-profile-name').text.strip()
            # Amazon's format is "x.0 stars out of 5" where x = # of stars
            rating = html_review.find('span', class_='a-icon-alt').text.strip()[0]
            review_body = html_review.find('span', attrs={'data-hook': 'review-body'}).text.strip()
            reviews.append({'customer_name': name, 'rating': int(rating), 'review': review_body})

        st.write("HTML Reviews (After):", html_reviews)
        return reviews

    def __get_reviews(self, asin: str, num_reviews: int, headless: bool = True):
        if num_reviews % 5 != 0:
            raise ValueError(f"num_reviews parameter provided, {num_reviews}, is not divisible by 5")

        base_url = self.__amazon_review_url + asin
        overall_reviews = []

        for star_num in range(1, 6):
            url = base_url + self.__star_page_suffix[star_num]
            st.write("URL:", url)
            page_number = 1
            reviews = []
            reviews_per_star = num_reviews // 5

            while len(reviews) <= reviews_per_star:
                page_url = url + str(page_number)

                # no reviews means we've exhausted all reviews
                page_reviews = self.__get_rated_reviews(page_url, headless)
                if not page_reviews:
                    break

                reviews += page_reviews
                page_number += 1

                # Add a 30-second delay after each request
                time.sleep(30)

            # shave off extra reviews coming from the last page
            reviews = reviews[:reviews_per_star]
            overall_reviews += reviews

        st.write("Overall Reviews:", overall_reviews)
        return overall_reviews

    def get_closest_product_reviews(self, search_query: str, num_reviews: int, headless: bool = True, debug: bool = False):
        if not search_query:
            raise ValueError('Search query provided is an empty string')

        if debug:
            start = time.time()

        html_page = self.__get_amazon_search_page(search_query, headless)
        st.write("HTML Response:", html_page)
        product_asin = self.__get_closest_product_asin(html_page)
        reviews = self.__get_reviews(asin=product_asin, num_reviews=num_reviews, headless=headless)

        if debug:
            end = time.time()
            print(f"{round(end - start, 2)} seconds taken")

        return reviews

    def get_product_reviews_by_asin(self, product_asin: str, num_reviews: int, headless: bool = False, debug: bool = False):
        if not product_asin:
            raise ValueError('ASIN provided is an empty string')

        if debug:
            start = time.time()

        reviews = self.__get_reviews(asin=product_asin, num_reviews=num_reviews, headless=headless)

        if debug:
            end = time.time()
            print(f"{round(end - start, 2)} seconds taken")

        return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='AmazonScraper',
        description='Fetch Amazon product reviews based on a search query or ASIN.'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--query', type=str, help='Product search query to fetch reviews for')
    group.add_argument('--asin', type=str, help='Product ASIN to fetch reviews for')
    parser.add_argument('num_reviews', type=int, help='Number of reviews to fetch')
    
    # optional flags
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode', default=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug output', default=False)

    args = parser.parse_args()

    scraper = AmazonScraper()
    
    if args.query:
        reviews = scraper.get_closest_product_reviews(
            search_query=args.query,
            num_reviews=args.num_reviews,
            headless=args.headless,
            debug=args.debug
        )
    elif args.asin:
        reviews = scraper.get_product_reviews_by_asin(
            product_asin=args.asin,
            num_reviews=args.num_reviews,
            headless=args.headless,
            debug=args.debug
        )

    for review in reviews:
        print(review, end='\n\n')
