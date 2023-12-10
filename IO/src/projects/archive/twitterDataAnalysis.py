# author           : Prateek
# email            : prateekpatel.in@gmail.com
# description      : Fetch what is trending in your location/any location on twitter

from twython import Twython
from collections import Counter
from geopy import Nominatim


# Note if running from binder the text file below will give an
try:
    with open("twitter_credentials.txt", "r") as f:
        line_text = [line.strip() for line in f]
except:
    print(
        "Note: The credentials file is not available on Git.\n"
        "If running the script on Binder, specify own app KEY and SECRET below "
        "and the code should run without any issue.\n"
    )
    print(
        "go to: https://developer.twitter.com/en/apps \nand create an app to generate KEY and SECRET\n"
    )
    print(
        "Binder will not store these keys and everything will be reset when the file is closed."
    )


CONSUMER_KEY = line_text[0]
CONSUMER_SECRET = line_text[1]


# ### Making a search query
#
# refer: https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets for formatting the search query and understanding results format.
#
# Max. num of results restricted to 100 per search query so we loop over many times and make the same query.
# But to avoid the results from repeating, we change the max_id of search results after each iteration


def find_tweets(
    search_string="",  # "" to search everything
    location_of_interest="London",
    radius_of_interest_in_km=20,
    num_tweets_to_fetch=1000,
    type_of_result="all",  # all, mixed, recent or popular
    c_key=CONSUMER_KEY,
    c_secret=CONSUMER_SECRET,
):

    # initialisation
    twitter = Twython(c_key, c_secret)

    tweets = []
    word_list = []
    hashtag_list = []
    retweet_count_list = []
    favorite_count_list = []
    tweet_url_list = []

    # Search area definition
    geolocator = Nominatim(user_agent="GoogleV3")
    location = geolocator.geocode(location_of_interest)
    geo_code = (
        str(location.latitude)
        + ","
        + str(location.longitude)
        + ","
        + str(radius_of_interest_in_km)
        + "km"
    )

    num_results_per_query = min([num_tweets_to_fetch, 100])

    # In case there aren't enough results for the search term
    max_attempts = max(50, num_tweets_to_fetch // num_results_per_query * 2)

    print("fetching...")
    for i in range(0, max_attempts):
        if num_tweets_to_fetch < len(tweets):
            break  # we got all the tweets we asked for ... !!

        # ----------------------------------------------------------------#
        # STEP 1: Query Twitter
        # STEP 2: Save the returned tweets
        # STEP 3: Get the next max_id
        # ----------------------------------------------------------------#

        # STEP 1: Query Twitter
        if 0 == i:
            # Query twitter for data.
            results = twitter.search(
                q=search_string,
                count=str(num_results_per_query),
                geocode=geo_code,
                result_type=type_of_result,
            )
        else:
            # After the first call we should have max_id from result of previous call. Pass it in query.
            results = twitter.search(
                q=search_string,
                count=str(num_results_per_query),
                geocode=geo_code,
                result_type=type_of_result,
                include_entities="true",
                max_id=next_max_id,
            )

        # STEP 2: Save the returned tweets
        for status in results["statuses"]:
            user = status["user"]["screen_name"].encode("utf-8")
            user = user.decode("utf-8")  # to convert the encoded byte type into string
            text = status["text"].encode("utf-8")
            text = text.decode("utf-8")  # to convert the encoded byte type into string
            for word in text.split():
                word_list.append(word)

                if word.startswith("#"):
                    hashtag_list.append(word)

            tweets.append(text)  # Keep track of number of tweets
            favorite_count_list.append(status["favorite_count"])
            retweet_count_list.append(status["retweet_count"])
            tweet_url_list.append(
                "https://twitter.com/i/web/status/" + status["id_str"]
            )

        # STEP 3: Get the next max_id
        try:
            # Parse the data returned to get max_id to be passed in consequent call.
            next_results_url_params = results["search_metadata"]["next_results"]
            next_max_id = next_results_url_params.split("max_id=")[1].split("&")[0]
        except:
            # No more next pages
            break

    print("...Done")
    return (
        tweets,
        hashtag_list,
        tweet_url_list,
        retweet_count_list,
        word_list,
        favorite_count_list,
    )


# ### Search for tweets



search_string = input("Search String:")  # "" to search everything
location_of_interest = "London"
radius_of_interest_in_km = 20
num_tweets_to_fetch = 1000
type_of_result = "all"  # all, mixed, recent or popular

# Search area definition
geolocator = Nominatim(user_agent="GoogleV3")
location = geolocator.geocode(location_of_interest)

tweets, hashtag_list, tweet_url_list, retweet_count_list, word_list, favorite_count_list = find_tweets(
    search_string,
    location_of_interest,
    radius_of_interest_in_km,
    num_tweets_to_fetch,
    type_of_result,
    c_key=CONSUMER_KEY,
    c_secret=CONSUMER_SECRET,
)


# ### Post Processing


print(location, "\n")
print("Number of tweets fetched:", len(tweets))

print("\n Top Hashtags:")
c = Counter(hashtag_list)
for tags, count in c.most_common(5):
    print(tags, count)

# print("\n Most common words:")
# c = Counter(word_list)
# for tags, count in c.most_common(6):
#     print(tags,count)

print("\n")
max_retweet_index = sorted(
    range(len(retweet_count_list)), key=lambda x: -retweet_count_list[x]
)[0]

most_retweeted = tweets[max_retweet_index]
max_retweet_count = retweet_count_list[max_retweet_index]
max_retweet_url = tweet_url_list[max_retweet_index]
print(
    "(most) Retweeted:",
    max_retweet_count,
    "\n",
    most_retweeted,
    "\n\n",
    "tweet link:",
    max_retweet_url,
)

print("\n")
max_favorite_index = sorted(
    range(len(favorite_count_list)), key=lambda x: -favorite_count_list[x]
)[0]
most_favorite = tweets[max_favorite_index]
max_favorite_count = favorite_count_list[max_favorite_index]
max_favorite_url = tweet_url_list[max_favorite_index]

print(
    "(most) Favorited:",
    max_favorite_count,
    "\n",
    most_favorite,
    "\n\n",
    "tweet link:",
    max_favorite_url,
)
