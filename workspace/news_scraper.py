import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse

class NewsScraper:
    def __init__(self, base_url, max_retries=3, initial_delay=1, backoff_factor=2):
        """
        Initialize the news scraper with configuration.

        Args:
            base_url (str): The URL of the news website to scrape.
            max_retries (int): Maximum number of retry attempts for failed requests.
            initial_delay (float): Initial delay in seconds for the first retry.
            backoff_factor (float): Multiplier for exponential backoff between retries.
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _is_valid_url(self, url):
        """
        Check if a URL is valid.

        Args:
            url (str): URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _get_retry_delay(self, retry_count):
        """
        Calculate the delay for the next retry using exponential backoff.

        Args:
            retry_count (int): Current retry attempt number.

        Returns:
            float: Delay in seconds before the next retry.
        """
        delay = self.initial_delay * (self.backoff_factor ** (retry_count - 1))
        # Add jitter to avoid thundering herd problem
        jitter = random.uniform(0.5, 1.5)
        return delay * jitter

    def fetch_page(self, url=None, retry_count=1):
        """
        Fetch a web page with retry logic and error handling.

        Args:
            url (str): URL to fetch. If None, uses the base_url.
            retry_count (int): Current retry attempt number.

        Returns:
            requests.Response: Response object if successful.

        Raises:
            requests.exceptions.RequestException: If all retry attempts fail.
        """
        if url is None:
            url = self.base_url

        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if retry_count <= self.max_retries:
                delay = self._get_retry_delay(retry_count)
                print(f"Attempt {retry_count} failed. Retrying in {delay:.2f} seconds... Error: {e}")
                time.sleep(delay)
                return self.fetch_page(url, retry_count + 1)
            else:
                raise requests.exceptions.RequestException(f"All {self.max_retries} retry attempts failed. Last error: {e}")

    def extract_headlines(self, html_content, headline_selectors):
        """
        Extract headlines from HTML content using specified CSS selectors.

        Args:
            html_content (str): HTML content to parse.
            headline_selectors (list): List of CSS selectors for headline elements.

        Returns:
            list: List of extracted headlines.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            headlines = []
            for selector in headline_selectors:
                elements = soup.select(selector)
                for element in elements:
                    headline = element.get_text(strip=True)
                    if headline:
                        headlines.append(headline)
            return headlines
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []

    def scrape_headlines(self, headline_selectors):
        """
        Scrape headlines from the news website.

        Args:
            headline_selectors (list): List of CSS selectors for headline elements.

        Returns:
            list: List of extracted headlines.
        """
        try:
            response = self.fetch_page()
            headlines = self.extract_headlines(response.text, headline_selectors)
            return headlines
        except Exception as e:
            print(f"Failed to scrape headlines: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Example: BBC News headlines
    BBC_NEWS_URL = "https://www.bbc.com/news"
    HEADLINE_SELECTORS = [
        'h3.gs-c-promo-heading__title',  # BBC News headline selector
        'a.gs-c-promo-heading',
        'h2.headline'
    ]

    scraper = NewsScraper(BBC_NEWS_URL)
    headlines = scraper.scrape_headlines(HEADLINE_SELECTORS)

    print("Extracted Headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")