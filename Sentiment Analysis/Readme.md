### Libraries:
* Pandas for dataframe manipulation.
* Matplotlib for data visualization.
* Nltk and textblob for text analysis.
* Re for regular expression.
* Tweepy for extracting tweets from Twitter API.

### Problem Statement: Build a Sentiment Analyzer Engine which finds out the sentiment of any given text.

**Sentiment Analysis** is a Natural Language Processing technique used to determine wheather data is positive, negative or neutral. Often performed on textual data. It basically studies the subjective information in an expression, i.e. opinions, emotions, appraisals, or attitude towards topic, person or entity. It is extremely useful in social media monitoring as it allows us to gain an overview of the wider public opinion behind certain topics. 

### Dataset: 
* The data in this project are tweets form Twitter API which are only in English language and retweets are flitered out to avoid the extraction of duplicate tweets.

### To scrape tweets from the Twitter API:
* **Create a Twitter Developer Account:** Visit the Twitter Developer Platform (https://developer.twitter.com/en) and sign up for a developer account. Once your account is approved, you'll be able to create an application and obtain the necessary API credentials.

* **Create a Twitter Application:** After logging in to the Twitter Developer Platform, create a new application. Provide the required information about your project, such as the name, description, and purpose. Once your application is created, you'll be able to generate API keys and access tokens.

* **Obtain API Credentials:** In your Twitter application's dashboard, navigate to the "Keys and tokens" tab. Here, you'll find your API key, API secret key, access token, and access token secret. These credentials will be needed to authenticate your requests to the Twitter API.

* **Choose a Programming Language and Library:** Decide on a programming language you're comfortable with. Here I went with Twitter API scraping with Python. Additionally, select a library that provides an interface to the Twitter API, I chose to go with Tweepy for Python.

* **Set up the Library:** Install the chosen library by following its installation instructions. For example, if like me you're also using Python and Tweepy, you can install it via pip: pip install tweepy. You may also need to import the library into your project.

* **Authenticate your Application:** Using the API credentials obtained in Step 3, initialize the library with the appropriate authentication. This typically involves providing your API key, API secret key, access token, and access token secret to the library's authentication method.

* **Define Search Parameters:** Decide on the criteria for the tweets you want to scrape, such as specific keywords, hashtags, user mentions, or geolocation. Specify these parameters in your code to tailor the search to your needs.

### Data Preprocessing:
### Data Cleaning: 
* Using regular expression to remove tags e.g. "@some_user", links form extracted tweets, numbers in the tweets. 
* Converting all words to lower case.
* Converting words into its base form(Lemmatization) using 'WordNetLemmatizer()'.
* Removing stop words.

### Building a Sentiment Analyzer Engine:
* Using the library TextBlob to find the sentiment of the tweets. Using the "sentiment" function of the TextBlob library to get the "polarity". Polarity lies between -1 to +1. The 'get_polarity()' funtion is used to get the polarity of each tweet and the depending upon the polarity value the tweet can be placed in any one of the 7 review category. 

### Visualizing Results:
* Using a pie chart to conclude the results about the sentiment of the tweets with a quick glance.
