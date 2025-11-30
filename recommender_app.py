import os

import pandas as pd
import streamlit as st
from surprise import Dataset, SVD, get_dataset_dir
from surprise.model_selection import train_test_split


# --- 1. Data loading helpers -------------------------------------------------


@st.cache_resource
def load_movie_titles() -> dict:
    """
    Load a mapping from MovieLens item IDs to human-readable movie titles.

    Returns
    -------
    dict
        Keys are item IDs as strings, values are movie titles.
    """
    base_dir = get_dataset_dir()

    # In the Surprise cache, ml-100k often lives under a nested ml-100k/ml-100k directory.
    ml_root = os.path.join(base_dir, "ml-100k", "ml-100k")
    item_file = os.path.join(ml_root, "u.item")

    if not os.path.exists(item_file):
        st.error(
            f"Movie title file not found at:\n{item_file}\n"
            "Please double-check your MovieLens 100k folder layout."
        )
        return {}

    titles_df = pd.read_csv(
        item_file,
        sep="|",
        header=None,
        engine="python",
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"],
    )

    # Surprise uses raw IDs as strings, so we normalize them here.
    return dict(zip(titles_df["item_id"].astype(str), titles_df["title"]))


@st.cache_resource
def train_svd_model():
    """
    Load the MovieLens 100k data and train an SVD recommender.

    Returns
    -------
    algo : surprise.SVD
        Trained SVD model.
    user_ids : list[str]
        List of raw user IDs seen during training.
    """
    # Built-in MovieLens 100k dataset bundled with Surprise
    data = Dataset.load_builtin("ml-100k")

    # Core implementation detail: keep the same split and random_state
    trainset, _ = train_test_split(data, test_size=0.25, random_state=42)

    algo = SVD()

    st.info("Training SVD model on the MovieLens 100k dataset...")
    algo.fit(trainset)
    st.success("Model training complete ✅")

    # Convert internal user IDs back to the original raw IDs
    user_ids = [
        algo.trainset.to_raw_uid(inner_uid)
        for inner_uid in algo.trainset.all_users()
    ]

    return algo, user_ids


# --- 2. Recommendation + inspection logic ------------------------------------


def get_top_n_recommendations(algo: SVD, user_id: str, n: int) -> list[tuple[str, float]]:
    """
    Compute top-N predicted ratings for items the user has not rated yet.

    Parameters
    ----------
    algo : surprise.SVD
        Trained SVD model.
    user_id : str
        Raw MovieLens user ID.
    n : int
        Number of recommendations to return.

    Returns
    -------
    list of (item_id, predicted_rating)
    """
    trainset = algo.trainset

    # Map external user ID to Surprise's internal index
    inner_uid = trainset.to_inner_uid(user_id)

    # Items already rated by this user (store only the inner IDs)
    rated_inner_iids = {inner_iid for (inner_iid, _) in trainset.ur[inner_uid]}

    # All items in the training set
    all_inner_iids = set(trainset.all_items())

    # Candidates = items the user has not rated
    candidate_inner_iids = all_inner_iids - rated_inner_iids

    predictions = []
    for inner_iid in candidate_inner_iids:
        raw_iid = trainset.to_raw_iid(inner_iid)
        # Core prediction call (unchanged)
        pred = algo.predict(user_id, raw_iid)
        predictions.append((pred.iid, pred.est))

    # Sort by predicted rating descending and keep the top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def build_recommendations_dataframe(
    recommendations: list[tuple[str, float]],
    title_lookup: dict,
) -> pd.DataFrame:
    """
    Turn a list of (item_id, prediction) into a tidy pandas DataFrame.
    """
    df = pd.DataFrame(recommendations, columns=["Item ID", "Predicted Rating"])
    df["Predicted Rating"] = df["Predicted Rating"].round(4)

    # Map IDs to movie titles; if a title is missing, keep the ID visible
    df["Movie Title"] = df["Item ID"].astype(str).map(title_lookup)
    df["Movie Title"].fillna(df["Item ID"], inplace=True)

    # Put human-friendly columns first
    df = df[["Movie Title", "Predicted Rating"]]

    # Rank for display
    df.index = df.index + 1
    df.index.name = "Rank"

    return df


# Helper to show what the user already likes
def build_user_history_dataframe(
    algo: SVD,
    user_id: str,
    title_lookup: dict,
    max_rows: int = 15,
) -> pd.DataFrame:
    """
    Build a DataFrame with a sample of movies already rated by the given user.

    This is mainly for HCI/UX: it helps the human user understand the model's
    "point of view" on that specific user.
    """
    trainset = algo.trainset
    inner_uid = trainset.to_inner_uid(user_id)

    # (item_inner_id, rating) pairs from the training data
    rated_pairs = trainset.ur[inner_uid]

    history = []
    for inner_iid, rating in rated_pairs[:max_rows]:
        raw_iid = trainset.to_raw_iid(inner_iid)
        title = title_lookup.get(str(raw_iid), str(raw_iid))
        history.append((title, raw_iid, rating))

    df = pd.DataFrame(history, columns=["Movie Title", "Item ID", "Original Rating"])
    df["Original Rating"] = df["Original Rating"].astype(float)

    return df


# --- 3. Streamlit UI ----------------------------------------------------------


def main():
    st.title("🎬 Movie Recommender System (SVD)")
    st.subheader("An AI/HCI Demo using Collaborative Filtering")
    st.markdown(
        "This small app uses the **MovieLens 100k** dataset and a classic "
        "**SVD-based collaborative filtering** model from the `surprise` library."
    )
    st.markdown("---")

    # Load model and metadata (cached across runs)
    algo, user_ids = train_svd_model()
    movie_titles = load_movie_titles()

    # Sidebar controls
    with st.sidebar:
        st.header("Recommendation Controls")

        selected_user_id = st.selectbox(
            "1. Pick a user ID:",
            user_ids,
            help="Each ID represents a real user from the MovieLens dataset.",
        )

        top_n = st.slider(
            "2. How many movie suggestions?",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
        )

        show_history = st.checkbox(
            "Show some movies this user already rated",
            value=False,
        )

        generate_button = st.button("🚀 Generate Recommendations")

    # Show the user's rating history (HCI/UX enhancement)
    if show_history:
        st.markdown("### What this user has already rated")
        history_df = build_user_history_dataframe(algo, selected_user_id, movie_titles)
        st.dataframe(history_df, use_container_width=True)

    if generate_button:
        st.markdown("---")
        st.header(f"Top {top_n} recommendations for user **{selected_user_id}**")

        # Core prediction logic
        top_recs = get_top_n_recommendations(algo, selected_user_id, top_n)
        recs_df = build_recommendations_dataframe(top_recs, movie_titles)

        st.dataframe(recs_df, use_container_width=True)
        st.caption(
            "Predicted ratings are generated by the SVD model based on similar user's "
            "preferences in the MovieLens 100k dataset."
        )

        st.markdown("---")
        st.success(
            "Titles have been mapped from raw item IDs for better readability and user experience."
        )


if __name__ == "__main__":
    main()