CANONICAL_CITIES: list[str] = [
    "NYC", "SF", "LA", "Chicago", "Boston", "DC",
]
NUMERICAL_COLS: list[str] = [
    "amenities", "accommodates", "bathrooms", "latitude", "longitude",
    "host_response_rate", "number_of_reviews", "review_scores_rating",
    "bedrooms", "beds",
    "first_review_month", "first_review_quarter",
    "days_since_first_review", "host_tenure_days", "is_holiday_season",
]
CATEGORICAL_COLS: list[str] = [
    "property_type", "room_type", "bed_type", "cancellation_policy",
    "cleaning_fee", "city", "host_identity_verified",
    "instant_bookable", "host_has_profile_pic",
]
PROPERTY_TYPE_CAT = [
    "Apartment", "House", "Condominium", "Townhouse", "Loft", "Other",
]
ROOM_TYPE_CAT = ["Entire home/apt", "Private room", "Shared room"]
BED_TYPE_CAT = ["Real Bed", "Futon", "Pull-out Sofa", "Airbed", "Couch"]
CANCELLATION_CAT = ["strict", "moderate", "flexible", "super_strict_30", "super_strict_60"]
CLEANING_FEE_CAT = ["True", "False"]
BOOLEAN_CAT = ["t", "f"]
