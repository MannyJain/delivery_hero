import random
import json
import uuid

NUM_RESTAURANTS = 50
ITEMS_PER_RESTAURANT = 20

locations = ["Varanasi", "Delhi", "Lucknow", "Mumbai", "Bangalore"]

cuisines = {
    "Indian": ["Biryani", "Paneer Curry", "Butter Chicken", "Dal Tadka", "Rogan Josh"],
    "Chinese": ["Hakka Noodles", "Fried Rice", "Chilli Chicken", "Manchurian"],
    "Italian": ["Margherita Pizza", "Pasta Alfredo", "Lasagna", "Bruschetta"],
    "Continental": ["Grilled Fish", "Caesar Salad", "Steak", "Roasted Chicken"],
    "South Indian": ["Masala Dosa", "Idli Sambhar", "Vada", "Uttapam"]
}

spice_levels = ["mild", "medium", "spicy"]
categories = ["Starter", "Main Course", "Dessert", "Beverage"]

restaurants = []
menu_items = []

item_counter = 1

for r_id in range(1, NUM_RESTAURANTS + 1):

    cuisine_type = random.choice(list(cuisines.keys()))
    rating = round(random.normalvariate(4.0, 0.4), 1)
    rating = max(3.0, min(rating, 4.8))

    popularity = int(rating * 20 + random.randint(0, 10))

    restaurant = {
        "restaurant_id": r_id,
        "restaurant_name": f"{cuisine_type} Delight {r_id}",
        "cuisine_type": cuisine_type,
        "average_rating": rating,
        "price_range": random.choice(["budget", "medium", "premium"]),
        "location": random.choice(locations),
        "delivery_time_minutes": random.randint(20, 50),
        "is_pure_veg": random.choice([True, False]),
        "popularity_score": popularity
    }

    restaurants.append(restaurant)

    for _ in range(ITEMS_PER_RESTAURANT):

        dish_name = random.choice(cuisines[cuisine_type])
        veg = random.choice([True, False]) if not restaurant["is_pure_veg"] else True
        price = random.randint(150, 600)

        description = f"{dish_name} prepared with authentic {cuisine_type} spices and fresh ingredients."

        item = {
            "item_id": item_counter,
            "restaurant_id": r_id,
            "item_name": dish_name,
            "description": description,
            "category": random.choice(categories),
            "price": price,
            "veg": veg,
            "spice_level": random.choice(spice_levels),
            "calories": random.randint(250, 900),
            "is_chef_special": random.random() < 0.1
        }

        menu_items.append(item)
        item_counter += 1

# Save files
with open("restaurants.json", "w") as f:
    json.dump(restaurants, f, indent=4)

with open("menu.json", "w") as f:
    json.dump(menu_items, f, indent=4)

print("Dataset generated successfully!")