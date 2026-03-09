import csv
import praw
from datetime import datetime, timezone, timedelta
from time import time
from typing import List, Dict, Any, Tuple, Optional

# CONFIG
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "StockMarket",
    "CryptoCurrency",
    "finance",
    "fintech",
]

LIMIT_POSTS_PER_SUB = 500
LIMIT_COMMENTS_PER_SUB = 5000

KEYWORDS = [
    "tsla", "tesla",
    "aapl", "apple",
    "goog", "googl", "google",
    "pltr", "palantir",
    "btc", "bitcoin",
    "eth", "ethereum",
]

WINDOW_MINUTES: Optional[int] = None  # e.g. 120 for last 2 hours

MAX_POST_TEXT_CHARS = 5000
MAX_COMMENT_TEXT_CHARS = 3000

OUT_POSTS_CSV = "reddit_posts.csv"
OUT_COMMENTS_CSV = "reddit_comments.csv"
OUT_POST_CONCAT_CSV = "reddit_postid_comments_concat.csv"

# load secrets from environment variables
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-scraper-script")

if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise ValueError(
        "Missing Reddit credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET as environment variables."
    )

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

SGT = timezone(timedelta(hours=8))

# helpers
def clamp_text(text: Optional[str], max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"

def extract_post_id(link_id: Optional[str]) -> Optional[str]:
    if not link_id:
        return None
    s = str(link_id)
    return s[3:] if s.startswith("t3_") else s

def parse_start_date_to_utc_ts(start_date_str: str) -> float:
    """
    start_date_str: 'YYYY-MM-DD'
    interprets as 00:00:00 UTC of that date.
    """
    dt = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)
    return dt.timestamp()

def keyword_match(text: str, keywords: List[str]) -> bool:
    """
    case-insensitive substring match.
    If keywords is empty, treat as match-all.
    """
    if not keywords:
        return True
    t = (text or "").lower()
    return any(k in t for k in keywords)

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"⚠️ No rows for {path}")
        return

    # union of keys across all rows (prevents ValueError)
    fieldnames = sorted({k for r in rows for k in r.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# scraper (filters by start_date + keywords)
def scrape_since_with_keywords(
    start_ts: float,
    keywords: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    posts: List[Dict[str, Any]] = []
    comments: List[Dict[str, Any]] = []

    seen_posts = set()
    seen_comments = set()

    for sub in SUBREDDITS:
        sr = reddit.subreddit(sub)

        # posts
        print(f"\nPosts: r/{sub}")
        for p in sr.new(limit=None):
            if p.id in seen_posts:
                continue

            if p.created_utc < start_ts:
                break  # stop once older than start

            full_text = f"{p.title} {p.selftext or ''}".strip()
            if not keyword_match(full_text, keywords):
                continue

            posts.append({
                "post_id": p.id,
                "subreddit": str(p.subreddit),
                "author": str(p.author) if p.author else None,
                "created_utc": p.created_utc,
                "created_sgt": datetime.fromtimestamp(p.created_utc, tz=SGT).isoformat(),
                "title": p.title,
                "selftext": clamp_text(p.selftext, MAX_POST_TEXT_CHARS),
                "combined_text": clamp_text(full_text, MAX_POST_TEXT_CHARS),
                "score": int(p.score),
                "upvote_ratio": getattr(p, "upvote_ratio", None),
                "num_comments": int(p.num_comments),
                "total_awards_received": int(getattr(p, "total_awards_received", 0) or 0),
                "permalink": f"https://reddit.com{p.permalink}",
                "url": getattr(p, "url", None),
                "over_18": bool(getattr(p, "over_18", False)),
                "stickied": bool(getattr(p, "stickied", False)),
                "locked": bool(getattr(p, "locked", False)),
            })
            seen_posts.add(p.id)

        # comments
        print(f"Comments: r/{sub}")
        for c in sr.comments(limit=None):
            if c.id in seen_comments:
                continue

            if c.created_utc < start_ts:
                break

            body = getattr(c, "body", "") or ""
            if not keyword_match(body, keywords):
                continue

            comments.append({
                "comment_id": c.id,
                "post_id": extract_post_id(getattr(c, "link_id", None)),
                "parent_id": getattr(c, "parent_id", None),
                "subreddit": str(c.subreddit),
                "author": str(c.author) if c.author else None,
                "created_utc": c.created_utc,
                "created_sgt": datetime.fromtimestamp(c.created_utc, tz=SGT).isoformat(),
                "body": clamp_text(body, MAX_COMMENT_TEXT_CHARS),
                "score": int(getattr(c, "score", 0) or 0),
                "total_awards_received": int(getattr(c, "total_awards_received", 0) or 0),
                "is_submitter": bool(getattr(c, "is_submitter", False)),
                "distinguished": getattr(c, "distinguished", None),
                "permalink": f"https://reddit.com{c.permalink}",
            })
            seen_comments.add(c.id)

    return posts, comments

# build post-level concat csv
def build_post_concat(posts: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comments_by_post: Dict[str, List[Dict[str, Any]]] = {}
    for c in comments:
        pid = c.get("post_id")
        if pid:
            comments_by_post.setdefault(pid, []).append(c)

    rows: List[Dict[str, Any]] = []
    for p in posts:
        pid = p["post_id"]
        post_comments = sorted(comments_by_post.get(pid, []), key=lambda x: x.get("created_utc") or 0)

        concat_comments = " ||| ".join(
            [c.get("body", "").strip() for c in post_comments if (c.get("body") or "").strip()]
        )

        rows.append({
            "post_id": pid,
            "post_text": p.get("combined_text", ""),
            "all_comments_concat": concat_comments,
        })

    return rows

# main
if __name__ == "__main__":
    print("=== Reddit Scraper (start_date + keyword filter) ===")

    start_date = input("Enter start_date (YYYY-MM-DD): ").strip()
    keywords_raw = input("Enter keywords/tickers (comma-separated): ").strip()

    keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if not keywords:
        print("⚠️ No keywords entered. Will scrape EVERYTHING since start_date (can be huge).")

    start_ts = parse_start_date_to_utc_ts(start_date)
    print(f"Start date UTC: {start_date} -> ts={start_ts}")
    print("Keywords:", keywords)

    posts, comments = scrape_since_with_keywords(start_ts, keywords)

    write_csv(OUT_POSTS_CSV, posts)
    write_csv(OUT_COMMENTS_CSV, comments)

    post_concat_rows = build_post_concat(posts, comments)
    write_csv(OUT_POST_CONCAT_CSV, post_concat_rows)

    print("\nDone.")
    print("Posts:", len(posts))
    print("Comments:", len(comments))
    print("Post-level concat rows:", len(post_concat_rows))
    print("Saved:", OUT_POSTS_CSV, OUT_COMMENTS_CSV, OUT_POST_CONCAT_CSV)