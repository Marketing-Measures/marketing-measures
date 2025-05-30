# %% [markdown]
# You are recommended to select `GPU` as the hardware accelerator. In Colab, navigate menu to `Runtime` -> `Change runtime type` and select `GPU`.

# %%
!pip install marketing_measures --no-deps

# %%
import pandas as pd

from marketing_measures import MarketingEmphasisScorer

# %%
scorer = MarketingEmphasisScorer(model_name="sentence-transformers/all-mpnet-base-v2")

# %%
scorer.get_model_info()["all_dimension_names_in_order"]

# %%
texts = [
    "The firm uses AI to understand customer needs and preferences.",
    "We continuously scan the market to gather insights about competitors.",
    "Accurate customer analysis helps refine our marketing strategies.",
    "Market research revealed a gap in services that we can exploit.",
    "Collecting smart data enables more precise targeting.",
    "Our marketing strategy incorporates both traditional and digital channels.",
    "The team designed a creative campaign to boost brand awareness.",
    "We follow a structured marketing planning process each quarter.",
    "Segmentation allows us to tailor messages for different customer groups.",
    "The campaign exemplified out-of-the-box thinking to reach Gen Z consumers.",
    "Marketing KPIs are monitored weekly to ensure implementation is on track.",
    "Fast execution of campaigns helps us stay ahead of competitors.",
    "The team was highly responsive to market changes during the rollout.",
    "Resources were reallocated flexibly to maximize performance.",
    "Speed and adaptability are essential for successful marketing implementation.",
    "Our pricing strategy reflects deep knowledge of competitor price points.",
    "Dynamic pricing tools support our price setting capabilities.",
    "The team reviewed pricing insights before launching the new service.",
    "We adjusted prices based on customer feedback and pricing analytics.",
    "A strong pricing capability ensures we capture value without losing volume.",
    "R&D investment led to a novel solution for customer pain points.",
    "The product development team worked closely with users to refine features.",
    "Test marketing revealed high acceptance for the innovation.",
    "Commercialization of the new product was accelerated by effective planning.",
    "Innovation management is central to our long-term competitiveness.",
    "Strong distributor relationships enhance our reach in remote markets.",
    "We provide ongoing retailer support to improve product visibility.",
    "Retailer cooperation was key in launching the new product line.",
    "Channel management focuses on adding value across the supply chain.",
    "Partnering with intermediaries allowed faster market penetration.",
    "The advertising campaign increased brand awareness significantly.",
    "Our marketing communication strategy emphasizes consistent messaging.",
    "Public relations helped manage reputation during the crisis.",
    "Image and branding efforts are supported by coordinated promotion.",
    "We rely on integrated marketing communication for maximum impact.",
    "Salespeople are trained regularly to improve selling skills.",
    "Sales support tools help enable more effective customer interactions.",
    "The sales strategy focuses on value-based selling.",
    "Sales management ensures quotas are aligned with overall goals.",
    "Sales controlling enables better forecasting and performance evaluation.",
]

# %%
results = scorer.score_texts(
    texts=texts,
    zca_transform="pre-trained",
    batch_size=32,
)

# %% [markdown]
# Results for marketing capabilities dimensions

# %%
df = pd.DataFrame(results)
marketing_capability_dimensions_list = [
    "marketing information management",
    "marketing planning capabilities",
    "marketing implementation capabilities",
    "pricing capabilities",
    "product development capabilities",
    "channel management",
    "marketing communication capabilities",
    "selling capabilities",
]

# %%
df[marketing_capability_dimensions_list]

# %%
df["text"] = texts

# %% [markdown]
# Top 3 sentences text for each dimension (only text)

# %%
for dimension in marketing_capability_dimensions_list:
    print(f"Top 3 sentences for {dimension}:")
    top_indices = df[dimension].nlargest(3).index
    for idx in top_indices:
        print(f"- {df['text'][idx]}")
    print("\n")
