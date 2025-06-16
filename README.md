# newsboat-recommender
Enjoy RSS feeds on newsboat with machine learning recommendations based on your interest!

# Instructions:
	1. Setup a virtual environment and install numpy and sklearn and sentence-transformers
	2. in newsboat, use ctrl+e to set a flag and over the time, set the flag to s for those articles which pique your interest 
       3. Once you have collected 200-300 articles suited to your interest, run python learn_preferences.py ~/.newsboat/cache.db (or any db file you use actively). This is a time consuming process if you have lots of articles in your database. The script will create a `models` directory for storing its output if it doesn't already exist.
        4. In your newsboat URL file, set up a query feed to filter the flags 'rec'. In this filter, your recommended
	   articles will pop up as unread.
	5. Once this is done, you can run generate_recommendations.py ~/.newsboat/cache.db to update your database with the recommendations.
	6. Then your recommendations will be in the filter set up in 4)


### Query Feed Setup
![image](query_feed_setup.png)


### Twitter RSS feed recommendation Generation
![image](twitter_recommendations.png)


### Thanks
Thanks to https://github.com/karpathy/arxiv-sanity-preserver. Most of the logic was taken from here.



## How the Machine Learning Works

When you first start exploring Newsboat Recommender it can feel almost magical: articles that you actually care about start appearing near the top of your feed, and you spend less time sifting through noise. The reality is that there is no magic involved. What drives these suggestions is a fairly simple yet carefully constructed machine learning pipeline. In this article we will dissect every moving part of the system. We will see how the data is gathered, how it is turned into numbers, and how a model learns from past reading habits to highlight the most relevant items in your feed. In true Jeremy Kun fashion, we will connect high level ideas to concrete code and mathematics, and by the end you will know exactly what is happening under the hood.

### The Data Source

Newsboat stores articles in an SQLite database. Every time you mark an item as read, the entry remains in the database along with metadata such as the feed URL, the author, the title, the article content, the time it was published, and any flags you might attach to it. The recommender treats this database as a user history. By default Newsboat lets you tag articles with one-letter flags. We rely on the `s` flag to mean "saved" or "interesting". When you press `Ctrl+e` on an entry and set it to `s`, you are telling the system, "This is the good stuff." After a few hundred items have been marked this way, the database provides a picture of your preferences.

The first script in the pipeline, `learn_preferences.py`, reads from this SQLite file and builds a training set. It fetches every article marked as read and inspects its flags. Items with the `s` flag become positive examples, while everything else becomes negative examples. From a machine learning standpoint, we are solving a binary classification problem: given an article, we want to predict whether you would have marked it with `s` or not.

### Turning Text into Vectors

Raw text must be translated into numbers before any learning can happen. The script uses two different approaches in tandem. The first is a classic bag-of-words technique called Term Frequency-Inverse Document Frequency (TF-IDF). The second uses a modern deep learning model from the sentence-transformers library. Each method captures a different notion of what an article is about.

Let us begin with TF-IDF. If you imagine each article as a bag containing its words, TF-IDF counts how often words appear in that bag, but it also discounts words that appear in almost every document. The idea is that ubiquitous words like "the" or "and" hold little information about what sets an article apart. So the system constructs a vocabulary from the corpus, picks out up to 5,000 distinct terms, and assigns each term an index. This is handled by scikit-learn's `TfidfVectorizer`. In the code, `generate_tfidf_pickles()` calls `v.fit(content_list)` to learn the vocabulary and `v.transform(content_list)` to convert every article into a sparse vector of term weights. The resulting matrix has shape `(number_of_articles, number_of_features)`.

This matrix is stored on disk so that future stages of the pipeline can reuse it. There is also a metadata pickle containing the vocabulary itself and a mapping from article IDs to row indices. While the output of `v.transform` is initially a compressed sparse matrix, the next step converts it to a dense floating-point array because the learning algorithm expects dense input.

### Smart Embeddings

Bag-of-words features are good at capturing the gist of longer texts, but they struggle with short fragments like article titles. To complement TF-IDF the code employs a sentence transformer—a pre-trained neural network that maps a string to a high-dimensional vector such that semantically similar strings lie close together. The specific model used here is `paraphrase-xlm-r-multilingual-v1`, which is multilingual and works entirely offline after being downloaded. To save memory and computation time the network is quantized using PyTorch's dynamic quantization capabilities. This converts the linear and embedding layers to 8-bit integer computations while keeping accuracy reasonably high.

During training, the titles of each article are fed into this neural network to produce what the code calls `X_smart`, a dense array where each row is a sentence embedding. These embeddings encode subtle notions of meaning. Articles that talk about similar topics or have similar phrasing in their titles end up with vectors pointing in roughly the same direction.

### Training the Models

With two sets of features in hand, the script trains two separate classifiers. Both are `LinearSVC` models from scikit-learn, which implement a Support Vector Machine with a linear kernel. SVMs aim to find a hyperplane that separates the positive examples from the negative ones with the largest possible margin. The choice of a linear kernel works well with high-dimensional input and is fast to train. One classifier is trained on the TF-IDF vectors (`clf` in the code) and the other on the sentence embeddings (`beclf`).

Why not combine the features into one big vector? It turns out that the scales and distributions of TF-IDF scores and neural embeddings are quite different. By training two independent models we can tune the contribution of each source of information later on. The training code sets a class weight of `balanced` to account for the fact that most articles are negatives—typically you mark far fewer items with `s` than you leave untagged. The `C` parameter is tuned lower for TF-IDF because those features are plentiful and tend to overfit otherwise.

### Storing the Model

Once training is complete, the pickled model file contains the two classifiers along with the name of the database they were derived from. Keeping track of the database matters because you might maintain multiple Newsboat feeds or start from scratch with a fresh database. The pickled metadata ensures that the subsequent recommendation step uses the right vocabulary and ID mappings.

### Generating Recommendations

The script `generate_recommendations.py` loads the same database but only looks at the most recent 200 articles. The assumption is that you may not want to revisit everything you have ever read—fresh content is usually more interesting. It loads the same metadata and vectorizer, then transforms the new articles into both TF-IDF vectors and title embeddings, exactly as was done during training.

Both classifiers, `clf` and `beclf`, compute decision scores for each article. A support vector machine's decision function essentially measures the distance from the separating hyperplane: positive scores mean the article is predicted to be "interesting" and negative scores mean the opposite. The code combines the two score arrays with a weighted average: 35% weight for TF-IDF (`s_tfidf * 0.35`) and 65% weight for the sentence embeddings (`s_smart * 0.65`). Through informal testing these weights offered good results, but you can tweak them to your liking.

After combining the scores, the script sorts the articles in descending order and selects the top 30 whose true label is negative—meaning you haven't already flagged them as interesting. This ensures that recommendations only include unseen or untagged items. The IDs of these articles are then written back into the database with a new flag `rec`, short for "recommendation." If you have set up a query feed in Newsboat to show all items with the flag `rec`, you will now see a freshly generated list of suggestions.

### Evaluating Effectiveness

Machine learning practitioners love to measure accuracy, precision, and recall. In an ideal world we would reserve a portion of the database for testing and compute metrics to quantify how well the model predicts your preferences. However, there is a subtle challenge: your interests change over time. Articles you enjoyed a year ago might no longer be relevant today. Cross-validation on a random split of the entire history could therefore paint a misleading picture.

Instead of formal accuracy scores, the effectiveness of this recommender is best judged by user satisfaction. After training the model on a few hundred labeled items, users typically report that a significant fraction of the top recommendations are indeed interesting. Because the system updates with every new batch of training data, it adapts to shifts in your preferences. If you start reading about a new topic and flag those articles with `s`, within a few training cycles the recommender picks up on the trend.

We can still reason about why the approach works. Both TF-IDF and sentence embeddings are high-dimensional representations that capture complementary aspects of article content. TF-IDF is excellent for picking up on keywords and short phrases that repeatedly show up in your favorite pieces. The sentence embeddings capture style and broader context, often doing a better job when the article has a creative title or when synonyms are involved. The SVM models, though simple, are effective at drawing a line that separates the two classes. Because SVMs focus on the points closest to the boundary—the so-called support vectors—they are robust even when the training set is imbalanced.

An interesting consequence of the design is that recommendations tend to be diverse. This stems from the weighting scheme in the combined score. The sentence embedding model might push content about related subjects to the top, while the TF-IDF model might favor pieces that share many of the same buzzwords as your favorites. Blending the two encourages variety without drifting too far off target.



### Combining the Scores

Historically the recommender simply averaged the scores of the two SVMs. The new
version fuses multiple rankings using **reciprocal rank fusion**. Each model
produces its own ranked list and the final score for an article is the sum of
`1/(k + rank)` terms. This approach favors items that appear near the top of any
list. In addition, the sentence embeddings are projected into a Poincar&eacute;
ball and compared to the average "interesting" article using a hyperbolic
distance. The resulting distance acts as a third ranking signal. RRF then
combines TF&#8209;IDF, embedding SVM, and hyperbolic similarity into a single
robust ordering.

### Running a Full Evaluation

To gauge how much the new fusion method helps, run `evaluate.py` after you have
trained a model. The script performs a small k-fold cross-validation on your
database and reports the mean average precision of both the old weighted-average
baseline and the fused ranking with hyperbolic distance.

```bash
python evaluate.py ~/.newsboat/cache.db
```

Comparing the two numbers reveals whether the fused recommender actually
outperforms the baseline on your data.

### Updating the Database

Once the recommendations are computed, `generate_recommendations.py` writes them back into the database via `update_newsboat_records`. This function executes an SQL `UPDATE` statement for each recommended ID, setting the flags column to `'rec'`. Newsboat's query feed mechanism can filter entries by flag, so you configure your feed list to show everything marked with `rec`. The next time you open Newsboat, the recommendations appear alongside the rest of your subscriptions as unread items.

Because the model writes directly to the same database that Newsboat uses, the whole process integrates smoothly with your existing workflow. There is no separate server or network call. Everything happens locally and offline after the initial download of the pre-trained sentence transformer.

### Interpreting the Results

Over time, as you continue to flag items with `s` and run the training script again, the SVMs incorporate the new examples. In practice users have reported that the top-ranked items often match their interests with surprising accuracy. Because the model is lightweight and the features are precomputed, the recommendation step takes only a few seconds even on modest hardware.

One insightful way to gauge performance is to look at the support vectors themselves. These are the training examples that lie closest to the decision boundary. In scikit-learn you can retrieve the indices of support vectors from a trained SVM. Inspecting those articles often reveals borderline cases: maybe you flagged them as interesting even though they were only tangentially related to your favorite topics. The model uses these borderline cases to fine tune the boundary, so reviewing them might help you adjust your own labeling strategy. If you want the recommendations to focus on a specific subject, make sure to consistently flag such articles with `s`.

### Why This Approach Works Well

There are many sophisticated recommender systems in the wild, from matrix factorization to deep learning with attention mechanisms. Why does this relatively simple combination of TF-IDF and SVM perform so well for personal RSS feeds? The key lies in the nature of the data. Your reading history is small compared to the datasets used by corporate recommendation engines. We do not have millions of examples to feed a deep neural network. But we do have the full text of each article, which is rich information. TF-IDF shines in this scenario because it can extract meaningful signals from as few as a couple hundred documents. The sentence embeddings add a semantic dimension that catches related topics even when the vocabulary does not match exactly.

Moreover, the problem is inherently personalized. The goal is not to guess what the average reader likes but what _you_ like. With a small number of classes and a simple model, we avoid overfitting to noise while still capturing strong patterns in your behavior.

### Limitations and Future Directions
Although the model works well in practice, it has limits. Because it treats each article independently, it lacks a notion of how your interests evolve across time or how feeds relate to one another. The vocabulary is fixed when you train, so new slang or topics may fall through the cracks until you retrain. A more advanced system might include user-specific metadata or contextual features from the feed itself. Online learning approaches could keep the model in sync with your habits as they shift, while larger neural models could capture deeper semantic nuances. Each of these improvements would come at the cost of more computation and complexity, but they offer promising avenues for exploration.



### Conclusion

We have walked through the entire lifecycle of an article recommendation, from recording your clicks in the Newsboat database to translating text into high-dimensional vectors and training two linear SVM models. We have seen how the combined score of these models identifies promising articles and how the system integrates seamlessly with Newsboat's flag-based workflow. While the approach is straightforward, it leverages powerful ideas from natural language processing and machine learning to tailor your feed to your interests. Most importantly, it remains transparent and customizable. You can inspect every step, tweak the weights, and even swap out components if you like. By demystifying the machinery, we hope you feel empowered to adapt and extend the system to fit your own reading habits.

