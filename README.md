# Airbnb_Review_Rating_Analysis 


## Introduction

This project investigates what separates 5-star Airbnb reviews from
lower-rated reviews. The data were scraped from Airbnb listings in
Manhattan, New York City. The dataset includes reviews from over 80
listings and contains over 3000 review texts with corresponding star
ratings.

The analysis combines text processing, sentiment analysis, topic
modeling, and text classification to examine linguistic and thematic
differences between 5-star and 1–4-star reviews. Instead of focusing on
ratings alone, the project analyzes review language to understand how
emotional tone, recurring themes, and specific operational issues relate
to rating outcomes.

By integrating these methods, the results provide structured insights
into what characterizes a “perfect” stay versus a “nearly perfect” one.
For hosts, the findings highlight recurring service gaps that may
prevent a listing from consistently receiving 5-star ratings. For
guests, the analysis suggests how review content can reveal meaningful
differences in experience that may not be fully captured by star ratings
alone.

## Scraping

``` python
import re
import pandas as pd
from playwright.async_api import async_playwright
```

``` python
search_url = "https://www.airbnb.com/s/Manhattan/homes"
base_url = "https://www.airbnb.com"
search_pages = 5
```

``` python
# Collect listing URLs from Airbnb search pages
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=False)
    page = await browser.new_page()

    # Open the search page
    await page.goto(search_url)
    await page.wait_for_timeout(2000)

    # Close any overlay that blocks clicks
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(800)

    listing_urls = []
    current_page_number = 1

    while current_page_number < search_pages + 1:
        # Close overlays again
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)

        # Collect all rooms links on the current results page
        room_links = page.locator('a[href*="/rooms/"]')
        room_links_count = await room_links.count()

        for i in range(room_links_count):
            href = await room_links.nth(i).get_attribute("href")
            if href and "/rooms/" in href:
                clean_href = href.split("?")[0]
                full_url = clean_href if clean_href.startswith("http") else base_url + clean_href
                listing_urls.append(full_url)

        # Deduplicate while keeping order
        listing_urls = list(dict.fromkeys(listing_urls))

        if current_page_number == search_pages:
            break

        # Go to the next page via the page number
        next_page_number = str(current_page_number + 1)

        await page.mouse.wheel(0, 8000)
        await page.wait_for_timeout(1200)

        pagination_nav = page.locator("nav").last
        await pagination_nav.get_by_text(next_page_number, exact=True).click()
        await page.wait_for_timeout(2500)

        # Close overlays after navigation
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(800)

        current_page_number += 1

    await browser.close()
```

``` python
# For each listing, open the reviews modal, scroll, and collect raw review blocks
scroll_steps = 120   # number of scroll rounds inside the review modal
stall_limit = 15     # stop if no new review id appears for N rounds

all_text_rows = []

async with async_playwright() as p:
    browser = await p.chromium.launch(headless=False)

    # Use a consistent locale
    context = await browser.new_context(
        locale="en-US",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
    )

    for listing_url in listing_urls:
        page = await context.new_page()

        try:
            # Open the listing page
            await page.goto(listing_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            # Close overlays
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)

            # Open the full reviews modal
            show_all_button = page.locator('button[data-testid="pdp-show-all-reviews-button"]').first
            if await show_all_button.count() == 0:
                await page.close()
                continue

            await show_all_button.click()
            await page.wait_for_timeout(1600)

            # The reviews are inside a dialog
            review_modal = page.get_by_role("dialog").first
            if await review_modal.count() == 0:
                await page.close()
                continue

            # Scrollable panel inside the dialog (fallback to modal if not found)
            scroll_panel = review_modal.locator('[data-testid="pdp-reviews-modal-scrollable-panel"]').first
            if await scroll_panel.count() == 0:
                scroll_panel = review_modal

            collected_review_ids = set()
            collected_texts = []
            stall_rounds = 0

            # Repeatedly collect new cards and scroll
            for scroll_round in range(scroll_steps):
                review_cards = review_modal.locator("[data-review-id]")
                card_count = await review_cards.count()

                new_in_this_round = 0

                for i in range(card_count):
                    card = review_cards.nth(i)
                    review_id = await card.get_attribute("data-review-id")

                    # Skip if the card has no id or was already collected
                    if not review_id or review_id in collected_review_ids:
                        continue

                    # Expand truncated text if "Show more" exists
                    more_button = card.get_by_role("button", name="Show more").first
                    if await more_button.count() > 0:
                        try:
                            await more_button.click(timeout=1500)
                            await page.wait_for_timeout(150)
                        except:
                            pass

                    review_text = await card.inner_text()
                    if review_text:
                        collected_review_ids.add(review_id)
                        collected_texts.append(review_text)
                        new_in_this_round += 1

                # If scrolling no longer loads new review cards, stop early
                if new_in_this_round == 0:
                    stall_rounds += 1
                else:
                    stall_rounds = 0

                if stall_rounds >= stall_limit:
                    break

                # Scroll down inside the modal
                await scroll_panel.evaluate("function(el) { el.scrollTop = el.scrollHeight; }")
                await page.wait_for_timeout(800)

            # Save raw blocks 
            for review_text in collected_texts:
                all_text_rows.append({"raw_text": review_text})

        except Exception:
            pass

        await page.close()

    await browser.close()
```

``` python
raw_df = pd.DataFrame(all_text_rows)
raw_df.to_csv("~/Desktop/Unstructured Data Analytics/final_project/airbnb_reviews_raw.csv", index=False)
```

``` python
import pandas as pd
df = pd.read_csv('~/Desktop/Unstructured Data Analytics/final_project/airbnb_reviews_raw.csv')
```

``` python
df["raw_text"] = df["raw_text"].astype(str)

# Extract star rating from the raw block
df["stars"] = df["raw_text"].str.extract(r"Rating,\s*([1-5])\s*stars?", expand=False)
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
```

``` python
# Extract guest review text
text = df["raw_text"].str.split("Stayed", n=1).str[-1] # keep everything after the first "Stayed"
text = text.str.split("\n", n=1).str[-1] # remove the the first line after "Stayed"
text = text.str.split("Response from", n=1).str[0] # remove host response if it exists

# final whitespace cleanup
df["review_text"] = (text.fillna("").str.replace(r"\s+", " ", regex=True).str.strip())
```

``` python
clean = df[["review_text", "stars"]].copy()

clean = clean[clean["review_text"] != ""]
clean = clean.dropna(subset=["stars"])
clean = clean.drop_duplicates().reset_index(drop=True)
```

``` python
print("Clean rows:", len(clean))
print(clean["stars"].value_counts().sort_index())
```

    Clean rows: 3333
    stars
    1      33
    2      28
    3     107
    4     370
    5    2795
    Name: count, dtype: int64

``` python
clean.to_csv("airbnb_reviews_clean.csv", index=False)
clean.head()
```

## Text Processing

``` python
import pandas as pd
import re

review = pd.read_csv('~/Desktop/Unstructured Data Analytics/final_project/airbnb_reviews_clean.csv')
```

### Writing Style Differences Between 5-Star and Lower-Rated Reviews

``` python
# Create group variable
review['group'] = review['stars'].apply(lambda x: '5-star' if x == 5 else '1-4-star')
```

``` python
review['review_text'] = review['review_text'].astype(str)

# Count total words
word_pat = re.compile(r"[A-Za-z']+")
review['word_count'] = review['review_text'].str.findall(word_pat).apply(len)

# Count sentences
review['sentence_count'] = review['review_text'].apply(lambda x: len([s for s in re.split(r'[.!?]+', x) if s.strip()]))

# Count exclamation marks
review['exclam_count'] = review['review_text'].apply(lambda x: x.count('!'))
```

``` python
# Compare writing style across groups
style_summary = review.groupby('group').agg(
    n=('review_text', 'count'),
    avg_word_count=('word_count', 'mean'),
    avg_sentence_count=('sentence_count', 'mean'),
    avg_exclam_count=('exclam_count', 'mean')
).round(2)

style_summary
```



|          | n    | avg_word_count | avg_sentence_count | avg_exclam_count |
|----------|------|----------------|--------------------|------------------|
| group    |      |                |                    |                  |
| 1-4-star | 538  | 61.88          | 4.53               | 0.27             |
| 5-star   | 2795 | 44.92          | 3.93               | 0.83             |



Lower-rated reviews are longer on average (more words and more
sentences). 5-star reviews are shorter but more expressive, using more
exclamation marks. This suggests that high ratings often come with
enthusiastic, emotional language, while lower ratings contain more
detailed descriptions.

### Prepare clean text for modeling

``` python
import spacy
nlp = spacy.load('en_core_web_lg') 

review['review_clean_base'] = (review['review_text'].str.lower().str.replace('[0-9]+', '', regex=True))

docs = list(nlp.pipe(review['review_clean_base']))

# Lemmatize and remove stopwords, punctuation, spaces
review['tokens_lemma'] = [
    [token.lemma_ for token in doc
     if not token.is_stop and not token.is_space and not token.is_punct]
    for doc in docs
]

review['clean_text'] = review['tokens_lemma'].apply(lambda x: ' '.join(x))

review[['review_text', 'clean_text']].head()
```



|  | review_text | clean_text |
|----|----|----|
| 0 | I’ve stayed in a hostel once before, and this ... | stay hostel like imagine hostel place wonderfu... |
| 1 | The people in the photo (hosts) are not the pe... | people photo host people see location site des... |
| 2 | this is my fifth time visiting manhattan and t... | fifth time visit manhattan definitely favorite... |
| 3 | I love this place! As newbie coming into the c... | love place newbie come city work trip hostel p... |
| 4 | I have never stayed in a hostel before so I wa... | stay hostel little nervous great experience ph... |



``` python
# Remove reviews that are too short
review['token_count'] = review['tokens_lemma'].apply(len)
review = review[review['token_count'] >= 5]
```

## Sentiment Analysis

``` python
from textblob import TextBlob

review['blob'] = review['review_text'].apply(lambda x: TextBlob(str(x)))

review['tb_polarity'] = review['blob'].apply(lambda x: x.sentiment.polarity)

review['tb_subjectivity'] = review['blob'].apply(lambda x: x.sentiment.subjectivity)
```

``` python
sentiment_summary = review.groupby('group').agg(
    avg_polarity=('tb_polarity', 'mean'),
    avg_subjectivity=('tb_subjectivity', 'mean')
).round(2)

sentiment_summary
```



|          | avg_polarity | avg_subjectivity |
|----------|--------------|------------------|
| group    |              |                  |
| 1-4-star | 0.21         | 0.56             |
| 5-star   | 0.40         | 0.62             |



``` python
review['is_negative'] = review['tb_polarity'].apply(lambda x: 1 if x < 0 else 0)

negative_rate = review.groupby('group')['is_negative'].mean().round(3)

negative_rate
```

    group
    1-4-star    0.098
    5-star      0.005
    Name: is_negative, dtype: float64

Sentiment analysis reveals a clear emotional difference between 5-star
and lower-rated reviews.

On average, 5-star reviews have a much higher polarity score (0.41
vs. 0.22), indicating substantially more positive emotional language.
They are also slightly more subjective, suggesting that guests express
stronger personal feelings in perfect ratings.

In contrast, lower-rated reviews are far more likely to contain negative
sentiment. While almost none of the 5-star reviews are negative (0.5%),
nearly 10% of lower-rated reviews fall below zero polarity. However,
this difference is not absolute, some 4-star reviews still contain
highly positive language, suggesting that ratings are influenced by more
than just emotional tone.

## Topic Modeling

``` python
import pandas as pd
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
```

``` python
review['clean_text'] = review['clean_text'].astype('str')
review_5 = review[review['group'] == '5-star'].copy()
review_low = review[review['group'] == '1-4-star'].copy()
```

### Top words for 5-star topics

``` python
vectorizer_5 = CountVectorizer()
X_5 = vectorizer_5.fit_transform(review_5['clean_text'])
vocab_5 = vectorizer_5.get_feature_names_out()

model_5 = lda.LDA(n_topics=3, n_iter=500, random_state=1)
model_5.fit(X_5)
```

    <lda.lda.LDA at 0x37463be00>

``` python
topic_word_5 = model_5.topic_word_
n_top_words = 10

for i, topic_dist in enumerate(topic_word_5):
    topic_words = np.array(vocab_5)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
```

    Topic 0: stay place great clean host recommend location easy bus
    Topic 1: great room stay location hotel clean time staff check
    Topic 2: stay place feel host recommend home comfortable clean original

Topic 0: Positive Stay Experience and Convenient Location

This topic includes words such as “stay”, “place”, “great”, “clean”,
“host”, and “recommend”, which reflect overall satisfaction with the
accommodation. It also contains “location”, “easy”, and “bus”,
suggesting convenience and accessibility. This theme reflects guests who
emphasize a smooth and enjoyable stay.

Topic 1: Clean Rooms and Professional Service

This topic includes words such as “room”, “location”, “hotel”, “clean”,
“staff”, “check”, and “time”. These words relate to room quality,
service interactions, and check-in experiences. This theme reflects
guests who evaluate comfort and operational aspects in a positive way.

Topic 2: Comfort, Hospitality, and Feeling at Home

This topic includes words such as “host”, “home”, “comfortable”,
“recommend”, and “feel”. These words emphasize emotional comfort and
personal hospitality. This theme reflects guests who describe their stay
as welcoming and home-like.

### Top words for lower-rated topics

``` python
vectorizer_low = CountVectorizer()
X_low = vectorizer_low.fit_transform(review_low['clean_text'])
vocab_low = vectorizer_low.get_feature_names_out()

model_low = lda.LDA(n_topics=3, n_iter=500, random_state=1)
model_low.fit(X_low)
```

    <lda.lda.LDA at 0x37468f890>

``` python
topic_word_low = model_low.topic_word_
n_top_words = 10

for i, topic_dist in enumerate(topic_word_low):
    topic_words = np.array(vocab_low)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
```

    Topic 0: location place good great stay room clean time hotel
    Topic 1: host check room stay hotel day night airbnb guest
    Topic 2: room bed work shower hotel small water issue bathroom

Topic 0: General Stay and Location Mentions

This topic includes words such as “location”, “stay”, “room”, “clean”,
and “hotel”. These are common descriptive terms that appear in both high
and low ratings. They are presented with less emotional intensity
compared to 5-star reviews.

Topic 1: Check-In and Guest Experience Issues

This topic includes words such as “host”, “check”, “night”, and “guest”.
These words suggest communication, coordination, and service-related
interactions. The presence of these terms indicates potential friction
during the stay process.

Topic 2: Bathroom, Water, and Room Problems

This topic includes words such as “shower”, “water”, “bathroom”,
“issue”, “small”, and “work”. These words point to functional and
operational problems. This theme reflects concrete complaints about
physical conditions of the accommodation.

### Thematic Comparison Between 5-Star and Lower-Rated Reviews

Topic modeling reveals a clear thematic contrast between 5-star and
lower-rated reviews.

The 5-star topics consistently emphasize overall satisfaction,
hospitality, comfort, and convenience. Words such as “great,”
“recommend,” “comfortable,” and “host” highlight emotional approval and
positive experiences. Even when discussing operational aspects like
rooms or check-in, the tone remains affirming.

In contrast, lower-rated reviews shift toward specific functional
details. Words such as “shower,” “water,” “bathroom,” “issue,” and
“small” indicate concrete operational problems. Rather than describing
the overall experience, these reviews focus on breakdowns in service or
physical conditions.

While both groups mention common features like “room” and “location,”
the difference lies in emphasis. 5-star reviews frame these elements
positively as part of a satisfying stay, whereas lower-rated reviews
highlight them in the context of limitations or issues.

## Text Classification

The review text is converted into TF-IDF features to represent each
review numerically. Logistic regression is then applied as a binary
classification model, allowing interpretation of how specific words
increase or decrease the probability of a 5-star rating.

``` python
# Create binary target: 1 = 5-star, 0 = non-5-star
review['y_5star'] = review['stars'].apply(lambda x: 1 if x == 5 else 0)

review['y_5star'].value_counts()
```

    y_5star
    1    2561
    0     501
    Name: count, dtype: int64

``` python
from sklearn.model_selection import train_test_split

X = review['clean_text'].astype('str')
y = review['y_5star']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)
```

``` python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into TF-IDF matrix
tfidf_vec = TfidfVectorizer()

X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

tfidf_tokens = tfidf_vec.get_feature_names_out()
```

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train logistic regression classifier
clf = LogisticRegression(max_iter=1000, random_state=1)
clf.fit(X_train_tfidf, y_train)

# Predictions
y_pred = clf.predict(X_test_tfidf)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```


    Accuracy Score: 0.8695
    Accuracy Percentage: 86.95%

    Detailed Classification Report:
                  precision    recall  f1-score   support

               0       0.83      0.25      0.38       100
               1       0.87      0.99      0.93       513

        accuracy                           0.87       613
       macro avg       0.85      0.62      0.66       613
    weighted avg       0.87      0.87      0.84       613

``` python
# Get model coefficients for each TF-IDF token
coef = clf.coef_[0]

coef_df = pd.DataFrame({
    'word': tfidf_tokens,
    'coef': coef
}).sort_values('coef', ascending=False)
```

``` python
# Top words predicting 5-star
top_pos = coef_df.head(15)
top_pos
```



|      | word        | coef     |
|------|-------------|----------|
| 3100 | stay        | 2.636265 |
| 1390 | great       | 2.437662 |
| 848  | definitely  | 2.359992 |
| 2641 | recommend   | 2.007922 |
| 3261 | thank       | 1.924717 |
| 133  | amazing     | 1.846073 |
| 1496 | highly      | 1.798997 |
| 615  | comfortable | 1.714558 |
| 2363 | perfect     | 1.695846 |
| 1479 | helpful     | 1.556767 |
| 3654 | wonderful   | 1.470721 |
| 1070 | enjoy       | 1.349051 |
| 1531 | host        | 1.327415 |
| 2163 | nice        | 1.292186 |
| 1123 | excellent   | 1.244923 |



``` python
# Top words predicting non-5-star
top_neg = coef_df.tail(15).sort_values('coef', ascending=True)
top_neg
```



|      | word        | coef      |
|------|-------------|-----------|
| 2980 | smell       | -1.728234 |
| 1372 | good        | -1.665388 |
| 1142 | expensive   | -1.645613 |
| 2784 | room        | -1.616019 |
| 1537 | hotel       | -1.591088 |
| 1224 | fine        | -1.526159 |
| 1045 | elevator    | -1.498498 |
| 2977 | small       | -1.485944 |
| 1247 | floor       | -1.459704 |
| 2717 | reservation | -1.332683 |
| 904  | dirty       | -1.328880 |
| 3563 | wall        | -1.328856 |
| 2346 | pay         | -1.313625 |
| 3659 | work        | -1.254808 |
| 2281 | overall     | -1.136863 |



### Model Interpretation

This model was not built simply to predict ratings, since guests already
provide star scores. Instead, it helps quantify which specific words and
issues are most strongly associated with a drop from 5 stars to lower
ratings.

The model achieved a test accuracy of 0.869, showing that review text
alone can strongly distinguish between 5-star and non-5-star reviews. It
performs especially well in identifying 5-star reviews, with a recall of
0.990, suggesting that enthusiastic reviews share consistent linguistic
patterns. In contrast, lower-rated reviews are more varied in language
and therefore harder to classify.

The words with the strongest positive coefficients, such as great,
definitely, recommend, perfect, comfortable, amazing, and thank, reflect
emotional intensity and strong endorsement. These reviews emphasize
satisfaction and appreciation. In comparison, the words most associated
with non-5-star reviews, including smell, small, elevator, dirty,
expensive, floor, and reservation, focus on concrete operational issues.
Many lower-rated reviews do not sound extremely negative, but instead
point to specific friction points or unmet expectations.

### Implications for Hosts and Guests

For hosts, the results suggest that ratings are not reduced because of
location or general experience, since those themes appear across both
groups. Ratings tend to drop when guests encounter tangible problems
such as odors, limited space, noise, or maintenance issues. If hosts
want to convert 4-star reviews into 5-star reviews, the most effective
strategy is not to highlight positive features more strongly, but to
eliminate recurring operational pain points that weaken an otherwise
good stay.

For guests, the results suggest that looking at review content in
addition to star ratings may provide a more complete picture of a
listing. A 4-star review that highlights a great location and
comfortable stay may still be a good choice for many travelers,
depending on their priorities. By paying attention to recurring themes
in the text, such as space, noise, or bathroom conditions, guests can
decide whether those issues matter to them personally rather than
relying only on the overall rating.
