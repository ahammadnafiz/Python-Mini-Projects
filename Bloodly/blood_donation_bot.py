import os
import logging
from datetime import datetime, timedelta
import re
from typing import List, Tuple
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ConversationHandler, CallbackContext
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import aiosqlite
from dotenv import load_dotenv


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('.env')
BOT_TOKEN = os.getenv('API')

# Define constants
MENU, LOCATION, BLOOD_TYPE, CONTACT, PROFILE, FIND, LOCATION_FIND, PROFILE_VIEW, UPDATE_PROFILE, UPDATE_NAME, UPDATE_BLOOD_TYPE, UPDATE_CONTACT, UPDATE_LAST_DONATION, UPDATE_LOCATION, UPDATE_AVAILABILITY = range(15)
RESULTS_PER_PAGE = 5
RATE_LIMIT_PERIOD = timedelta(minutes=1)
MAX_REQUESTS = 6

REMINDER_RADIUS = 10  # km
REMINDER_COOLDOWN = timedelta(hours=24)  # How often a donor can receive reminders


BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
# Blood type compatibility chart
COMPATIBLE_BLOOD_TYPES = {
    'A+': ['A+', 'A-', 'O+', 'O-'],
    'A-': ['A-', 'O-'],
    'B+': ['B+', 'B-', 'O+', 'O-'],
    'B-': ['B-', 'O-'],
    'AB+': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'],
    'AB-': ['A-', 'B-', 'AB-', 'O-'],
    'O+': ['O+', 'O-'],
    'O-': ['O-']
}

# Rate limiting
rate_limit_dict = defaultdict(list)
# Initialize the geolocator
geolocator = Nominatim(user_agent="blood_donation_bot")

# Database setup
DB_PATH = 'blood_donation.db'

async def setup_database():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Check if the 'available' column exists
            async with db.execute("PRAGMA table_info(donors)") as cursor:
                columns = await cursor.fetchall()
                column_names = [column[1] for column in columns]
            
            if 'available' not in column_names:
                # Add the 'available' column if it doesn't exist
                await db.execute("ALTER TABLE donors ADD COLUMN available BOOLEAN NOT NULL DEFAULT TRUE")
                await db.commit()
                logger.info("Added 'available' column to donors table")
            
            # Ensure all other required columns exist
            await db.execute("""
            CREATE TABLE IF NOT EXISTS donors (
                id INTEGER PRIMARY KEY,
                user_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                blood_type TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                contact TEXT NOT NULL,
                last_donation TEXT,
                available BOOLEAN NOT NULL DEFAULT TRUE
            )
            """)
            
            # Create the donor_reminders table
            await db.execute("""
            CREATE TABLE IF NOT EXISTS donor_reminders (
                id INTEGER PRIMARY KEY,
                donor_id INTEGER,
                last_reminded TIMESTAMP,
                FOREIGN KEY (donor_id) REFERENCES donors (id)
            )
            """)
            await db.commit()
    except Exception as e:
        logger.error(f"Database setup error: {e}")

def parse_dms_coordinate(coord_str):
    decimal_pattern = r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)"
    decimal_match = re.match(decimal_pattern, coord_str)
    if decimal_match:
        return float(decimal_match.group(1)), float(decimal_match.group(2))

    # If not decimal, try parsing as DMS
    dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS])\s*(\d+)°(\d+)'([\d.]+)\"([EW])"
    dms_match = re.match(dms_pattern, coord_str)
    if dms_match:
        lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = dms_match.groups()
        lat = (float(lat_d) + float(lat_m)/60 + float(lat_s)/3600) * (1 if lat_dir == 'N' else -1)
        lon = (float(lon_d) + float(lon_m)/60 + float(lon_s)/3600) * (1 if lon_dir == 'E' else -1)
        return lat, lon

    return None

def check_rate_limit(user_id: int) -> bool:
    current_time = datetime.now()
    user_requests = rate_limit_dict[user_id]
    
    # Remove old requests
    user_requests = [time for time in user_requests if current_time - time < RATE_LIMIT_PERIOD]
    
    if len(user_requests) >= MAX_REQUESTS:
        return False
    
    user_requests.append(current_time)
    rate_limit_dict[user_id] = user_requests
    return True

def is_within_radius(donor_lat, donor_lon, request_lat, request_lon, radius_km):
    return geodesic((donor_lat, donor_lon), (request_lat, request_lon)).km <= radius_km

async def send_reminders(bot: Bot, blood_type: str, latitude: float, longitude: float):
    async with aiosqlite.connect(DB_PATH) as db:
        query = """
        SELECT d.id, d.user_id, d.blood_type, d.latitude, d.longitude, dr.last_reminded
        FROM donors d
        LEFT JOIN donor_reminders dr ON d.id = dr.donor_id
        WHERE d.blood_type = ? AND d.available = TRUE
        """
        async with db.execute(query, (blood_type,)) as cursor:
            donors = await cursor.fetchall()

        now = datetime.now()
        for donor in donors:
            if is_within_radius(donor[3], donor[4], latitude, longitude, REMINDER_RADIUS):
                if donor[5] is None or now - datetime.fromisoformat(donor[5]) > REMINDER_COOLDOWN:
                    try:
                        await bot.send_message(
                            chat_id=donor[1],
                            text=f"🚨 Blood Needed! 🚨\n\nSomeone in your area needs {blood_type} blood. If you're available to donate, please check the app for more details."
                        )
                        await db.execute(
                            "INSERT OR REPLACE INTO donor_reminders (donor_id, last_reminded) VALUES (?, ?)",
                            (donor[0], now.isoformat())
                        )
                    except Exception as e:
                        logger.error(f"Failed to send reminder to donor {donor[1]}: {e}")
        await db.commit()

async def start(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT * FROM donors WHERE user_id = ?", (user_id,))
        user_is_registered = await cursor.fetchone() is not None

    # Send the intro text
    await update.message.reply_text('''
                                    🩸🇧🇩 Welcome to BloodHeroes Bangladesh! 🇧🇩🩸

Assalamualaikum, future lifesaver! 🦸‍♀️🦸‍♂️

You've just taken the first heroic step towards becoming a real-life superhero. Your mission, should you choose to accept it, is to save lives with the power of your blood! 💪

🚀 Quick Start Guide:
• Type /menu to view all options
• Type /help for assistance anytime

Remember: Your blood type might be rare, but your kindness is legendary! One donation can save up to three lives. 😇

Fun Fact: Did you know? The "lucky" blood type in Bangladesh is B+, shared by about 35% of the population!

Ready to be someone's lifeline? Let's get started! Type /help to begin your heroic journey. 🎉

Made with ❤️ by 💻 Ahammad Nafiz. Check out more cool projects at github.com/ahammadnafiz!

#EveryDropCounts #BloodHeroesBangladesh
                                    ''')

    # Then send the menu options as before
    keyboard = [
        [InlineKeyboardButton("Donate Blood 🩸", callback_data='donate'),
         InlineKeyboardButton("Find Blood 🔍", callback_data='find')],
    ]
    if user_is_registered:
        keyboard.append([InlineKeyboardButton("My Profile 👤", callback_data='profile')])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        'What would you like to do?',
        reply_markup=reply_markup
    )
    return MENU

async def menu_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == 'donate':
        await query.message.reply_text('Please share your location. You can send your current location, a Google Maps link, or type an address.')
        return LOCATION
    elif query.data == 'find':
        return await find_blood(update, context)
    elif query.data == 'profile':
        return await show_profile(update, context)
    elif query.data == 'update_profile':
        return await update_profile_prompt(update, context)
    elif query.data == 'back_to_menu':
        return await back_to_menu(update, context)
    else:
        await query.message.reply_text('Invalid choice. Please select from the options provided.')
        return MENU

async def update_profile_prompt(update: Update, context: CallbackContext) -> int:
    update_text = ("What would you like to update?\n\n"
                   "1. Name\n"
                   "2. Blood Type\n"
                   "3. Contact Number\n"
                   "4. Last Donation Date\n"
                   "5. Location\n"
                   "6. Availability\n\n"
                   "Please send the number corresponding to your choice.")
    
    await update.callback_query.message.reply_text(update_text)
    return UPDATE_PROFILE

async def update_profile(update: Update, context: CallbackContext) -> int:
    choice = update.message.text.strip()

    if choice == '1':
        await update.message.reply_text('Please enter your new name:')
        return UPDATE_NAME
    elif choice == '2':
        keyboard = [[InlineKeyboardButton(bt, callback_data=f'update_blood_{bt}') for bt in BLOOD_TYPES[i:i+2]] for i in range(0, len(BLOOD_TYPES), 2)]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Please select your new blood type:', reply_markup=reply_markup)
        return UPDATE_BLOOD_TYPE
    elif choice == '3':
        await update.message.reply_text('Please enter your new contact number:')
        return UPDATE_CONTACT
    elif choice == '4':
        await update.message.reply_text('Please enter your last donation date (YYYY-MM-DD or "Never"):')
        return UPDATE_LAST_DONATION
    elif choice == '5':
        await update.message.reply_text('Please share your new location. You can send your current location, a Google Maps link, or type an address.')
        return UPDATE_LOCATION
    elif choice == '6':
        await update.message.reply_text('Are you available to donate? Please type "Yes" or "No".')
        return UPDATE_AVAILABILITY
    else:
        await update.message.reply_text('Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.')
        return UPDATE_PROFILE

async def update_location(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    user_id = user.id

    if update.message.location:
        latitude = update.message.location.latitude
        longitude = update.message.location.longitude
    elif update.message.text:
        coords = parse_dms_coordinate(update.message.text)
        if coords:
            latitude, longitude = coords
        else:
            coords = extract_coords_from_google_maps_link(update.message.text)
            if coords:
                latitude, longitude = coords
            else:
                try:
                    location = geolocator.geocode(f"{update.message.text}, Bangladesh")
                    if location:
                        latitude = location.latitude
                        longitude = location.longitude
                    else:
                        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
                        return UPDATE_LOCATION
                except Exception as e:
                    logger.error(f"Geocoding error: {e}")
                    await update.message.reply_text('An error occurred while processing your location. Please try again.')
                    return UPDATE_LOCATION
    else:
        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
        return UPDATE_LOCATION

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE donors SET latitude = ?, longitude = ? WHERE user_id = ?", (latitude, longitude, user_id))
            await db.commit()
        await update.message.reply_text('Your location has been updated successfully.')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text('An error occurred while updating your location. Please try again later.')

    return ConversationHandler.END

# New function to handle name updates
async def update_name(update: Update, context: CallbackContext) -> int:
    new_name = update.message.text.strip()
    user_id = update.effective_user.id

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE donors SET name = ? WHERE user_id = ?", (new_name, user_id))
            await db.commit()
        await update.message.reply_text('Your name has been updated successfully.')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text('An error occurred while updating your name. Please try again later.')

    return ConversationHandler.END  # Return this to end the conversation or another valid state

async def update_blood_type_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    new_blood_type = query.data.split('_')[2]
    user_id = update.effective_user.id

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE donors SET blood_type = ? WHERE user_id = ?", (new_blood_type, user_id))
            await db.commit()
        await query.message.reply_text(f'Your blood type has been updated to {new_blood_type}.')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await query.message.reply_text('An error occurred while updating your blood type. Please try again later.')

    return ConversationHandler.END

async def update_contact(update: Update, context: CallbackContext) -> int:
    new_contact = update.message.text.strip()
    user_id = update.effective_user.id

    if re.match(r'^\+?880\d{10}$', new_contact):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE donors SET contact = ? WHERE user_id = ?", (new_contact, user_id))
                await db.commit()
            await update.message.reply_text('Your contact number has been updated successfully.')
        except Exception as e:
            logger.error(f"Database error: {e}")
            await update.message.reply_text('An error occurred while updating your contact number. Please try again later.')
    else:
        await update.message.reply_text('Invalid contact number. Please enter a valid Bangladesh phone number.')
        return UPDATE_CONTACT

    return ConversationHandler.END

async def update_last_donation(update: Update, context: CallbackContext) -> int:
    new_last_donation = update.message.text.strip()
    user_id = update.effective_user.id

    if new_last_donation.lower() == 'never':
        new_last_donation = None
    else:
        try:
            new_last_donation = datetime.strptime(new_last_donation, '%Y-%m-%d').date().isoformat()
        except ValueError:
            await update.message.reply_text('Invalid date format. Please use YYYY-MM-DD or "Never".')
            return UPDATE_LAST_DONATION

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE donors SET last_donation = ? WHERE user_id = ?", (new_last_donation, user_id))
            await db.commit()
        await update.message.reply_text('Your last donation date has been updated successfully.')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text('An error occurred while updating your last donation date. Please try again later.')

    return ConversationHandler.END

async def update_availability(update: Update, context: CallbackContext) -> int:
    response = update.message.text.strip().lower()
    user_id = update.effective_user.id

    if response in ['yes', 'no']:
        new_availability = response == 'yes'
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE donors SET available = ? WHERE user_id = ?", (new_availability, user_id))
                await db.commit()
            await update.message.reply_text(f'Your availability has been updated to {"Available" if new_availability else "Not Available"}.')
        except Exception as e:
            logger.error(f"Database error: {e}")
            await update.message.reply_text('An error occurred while updating your availability. Please try again later.')
    else:
        await update.message.reply_text('Invalid input. Please type "Yes" or "No".')
        return UPDATE_AVAILABILITY

    return ConversationHandler.END

async def location(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user

    if update.message.location:
        context.user_data['latitude'] = update.message.location.latitude
        context.user_data['longitude'] = update.message.location.longitude
    elif update.message.text:
        coords = parse_dms_coordinate(update.message.text)
        if coords:
            context.user_data['latitude'], context.user_data['longitude'] = coords
        else:
            coords = extract_coords_from_google_maps_link(update.message.text)
            if coords:
                context.user_data['latitude'], context.user_data['longitude'] = coords
            else:
                try:
                    location = geolocator.geocode(f"{update.message.text}, Bangladesh")
                    if location:
                        context.user_data['latitude'] = location.latitude
                        context.user_data['longitude'] = location.longitude
                    else:
                        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
                        return LOCATION
                except Exception as e:
                    logger.error(f"Geocoding error: {e}")
                    await update.message.reply_text('An error occurred while processing your location. Please try again.')
                    return LOCATION
    else:
        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
        return LOCATION

    keyboard = [[InlineKeyboardButton(bt, callback_data=f'blood_{bt}') for bt in BLOOD_TYPES[i:i+2]] for i in range(0, len(BLOOD_TYPES), 2)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('What is your blood type?', reply_markup=reply_markup)
    return BLOOD_TYPE

def extract_coords_from_google_maps_link(link):
    patterns = [
        r"@(-?\d+\.\d+),(-?\d+\.\d+)",
        r"ll=(-?\d+\.\d+),(-?\d+\.\d+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None

async def blood_type_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    blood_type = query.data.split('_')[1]
    context.user_data['blood_type'] = blood_type
    await query.message.reply_text('Please provide your contact number.')
    return CONTACT

async def contact(update: Update, context: CallbackContext) -> int:
    contact = update.message.text

    if re.match(r'^\+?880\d{10}$', contact):
        context.user_data['contact'] = contact
        await update.message.reply_text('When was your last blood donation? (YYYY-MM-DD or "Never")')
        return PROFILE
    else:
        await update.message.reply_text('Invalid contact number. Please enter a valid Bangladesh phone number.')
        return CONTACT

async def profile(update: Update, context: CallbackContext) -> int:
    last_donation = update.message.text

    if last_donation.lower() == 'never':
        last_donation = None
    else:
        try:
            last_donation = datetime.strptime(last_donation, '%Y-%m-%d').date().isoformat()
        except ValueError:
            await update.message.reply_text('Invalid date format. Please use YYYY-MM-DD or "Never".')
            return PROFILE

    user_id = update.effective_user.id
    name = update.effective_user.full_name

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
            INSERT OR REPLACE INTO donors
            (user_id, name, blood_type, latitude, longitude, contact, last_donation, available)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name, context.user_data['blood_type'],
                  context.user_data['latitude'], context.user_data['longitude'],
                  context.user_data['contact'], last_donation, True))
            await db.commit()

        await update.message.reply_text('Thank you for registering as a donor! 🎉\n\nYour information has been saved successfully. By default, your availability status is set to "Available".')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text('An error occurred while saving your information. Please try again later.')

    return ConversationHandler.END

async def find_nearest_donors(lat: float, lon: float, blood_type: str, page: int = 1, limit: int = RESULTS_PER_PAGE) -> Tuple[List[Tuple], int]:
    logger.info(f"Finding nearest donors to location: ({lat}, {lon}) for blood type: {blood_type}")

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Check if 'available' column exists
            async with db.execute("PRAGMA table_info(donors)") as cursor:
                columns = await cursor.fetchall()
                column_names = [column[1] for column in columns]
            
            if 'available' in column_names:
                query = "SELECT * FROM donors WHERE blood_type = ? AND available = TRUE"
            else:
                query = "SELECT * FROM donors WHERE blood_type = ?"
            
            params = [blood_type]

            async with db.execute(query, params) as cursor:
                exact_donors = await cursor.fetchall()

            if not exact_donors:
                compatible_types = COMPATIBLE_BLOOD_TYPES[blood_type]
                placeholders = ','.join('?' for _ in compatible_types)
                
                if 'available' in column_names:
                    query = f"SELECT * FROM donors WHERE blood_type IN ({placeholders}) AND available = TRUE"
                else:
                    query = f"SELECT * FROM donors WHERE blood_type IN ({placeholders})"
                
                params = compatible_types

                async with db.execute(query, params) as cursor:
                    compatible_donors = await cursor.fetchall()

                donors = compatible_donors
            else:
                donors = exact_donors

            if not donors:
                logger.info(f"No available donors found with blood type: {blood_type} or compatible types")
                return [], 0

            columns = ['id', 'user_id', 'name', 'blood_type', 'latitude', 'longitude', 'contact', 'last_donation']
            if 'available' in column_names:
                columns.append('available')
            
            df = pd.DataFrame(donors, columns=columns)

            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])

            df['distance'] = df.apply(lambda row: geodesic((lat, lon), (row['latitude'], row['longitude'])).km, axis=1)
            df_sorted = df.sort_values(by='distance')

            total_results = len(df_sorted)
            start_index = (page - 1) * limit
            end_index = start_index + limit

            nearest_donors = df_sorted.iloc[start_index:end_index]

            return [(row['blood_type'], row['distance'], row['contact'], row['latitude'], row['longitude']) for _, row in nearest_donors.iterrows()], total_results

    except Exception as e:
        logger.error(f"Error while finding nearest donors: {e}")
        return [], 0

async def find_blood(update: Update, context: CallbackContext) -> int:
    keyboard = [[InlineKeyboardButton(bt, callback_data=f'find_{bt}') for bt in BLOOD_TYPES[i:i+2]] for i in range(0, len(BLOOD_TYPES), 2)]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text('What blood type do you need?', reply_markup=reply_markup)
    return FIND

async def blood_type_find_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    blood_type = query.data.split('_')[1]
    context.user_data['needed_blood_type'] = blood_type
    await query.message.reply_text('Please enter the location where you need blood (district, address, or Google Maps link).')
    return LOCATION_FIND

async def handle_find_blood_location(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    if not check_rate_limit(user_id):
        await update.message.reply_text("You've reached the maximum number of requests. Please try again later.")
        return ConversationHandler.END

    if update.message.location:
        latitude = update.message.location.latitude
        longitude = update.message.location.longitude
    elif update.message.text:
        coords = parse_dms_coordinate(update.message.text)
        if coords:
            latitude, longitude = coords
        else:
            coords = extract_coords_from_google_maps_link(update.message.text)
            if coords:
                latitude, longitude = coords
            else:
                try:
                    location = geolocator.geocode(f"{update.message.text}, Bangladesh")
                    if location:
                        latitude = location.latitude
                        longitude = location.longitude
                    else:
                        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
                        return LOCATION_FIND
                except Exception as e:
                    logger.error(f"Geocoding error: {e}")
                    await update.message.reply_text('An error occurred while processing your location. Please try again.')
                    return LOCATION_FIND
    else:
        await update.message.reply_text('Invalid location. Please send a Google Maps link, your current location, or a specific address.')
        return LOCATION_FIND

    needed_blood_type = context.user_data.get('needed_blood_type')
    nearest_donors, total_results = await find_nearest_donors(latitude, longitude, blood_type=needed_blood_type, page=1)

    if not nearest_donors:
        await update.message.reply_text(f"No donors with blood type {needed_blood_type} or compatible types found in the area. Please try a wider search area.")
    else:
        if nearest_donors[0][0] == needed_blood_type:
            response = f"Found {total_results} donors with exact blood type match {needed_blood_type}:\n\n"
        else:
            response = f"No exact matches found. Showing {total_results} donors with compatible blood types for {needed_blood_type}:\n\n"

        for i, (blood_type, distance, contact, donor_lat, donor_lon) in enumerate(nearest_donors, 1):
            response += f"{i}. Blood Type: {blood_type}\n"
            response += f"   Distance: {distance:.2f} km\n"
            response += f"   Contact: {contact}\n"
            response += f"   Location: https://www.google.com/maps?q={donor_lat},{donor_lon}\n\n"

        if total_results > RESULTS_PER_PAGE:
            keyboard = [
                [InlineKeyboardButton("Next Page", callback_data=f"next_page_{2}_{latitude}_{longitude}_{needed_blood_type}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(response, reply_markup=reply_markup)
        else:
            await update.message.reply_text(response)

        # Send reminders to eligible donors
        await send_reminders(context.bot, needed_blood_type, latitude, longitude)

    return ConversationHandler.END

async def paginate_results(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    if not check_rate_limit(user_id):
        await query.message.reply_text("You've reached the maximum number of requests. Please try again later.")
        return

    _, page, lat, lon, blood_type = query.data.split('_')
    page = int(page)
    lat = float(lat)
    lon = float(lon)

    nearest_donors, total_results = await find_nearest_donors(lat, lon, blood_type=blood_type, page=page)

    if not nearest_donors:
        await query.message.reply_text("No more results found.")
        return

    response = f"Page {page} of donors for blood type {blood_type}:\n\n"
    for i, (blood_type, distance, contact, donor_lat, donor_lon) in enumerate(nearest_donors, 1):
        response += f"{i}. Blood Type: {blood_type}\n"
        response += f"   Distance: {distance:.2f} km\n"
        response += f"   Contact: {contact}\n"
        response += f"   Location: https://www.google.com/maps?q={donor_lat},{donor_lon}\n\n"

    keyboard = []
    if page > 1:
        keyboard.append(InlineKeyboardButton("Previous Page", callback_data=f"next_page_{page-1}_{lat}_{lon}_{blood_type}"))
    if page * RESULTS_PER_PAGE < total_results:
        keyboard.append(InlineKeyboardButton("Next Page", callback_data=f"next_page_{page+1}_{lat}_{lon}_{blood_type}"))

    reply_markup = InlineKeyboardMarkup([keyboard])
    await query.message.edit_text(response, reply_markup=reply_markup)

async def emergency_callback(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    blood_type = query.data.split('_')[1]
    context.user_data['emergency_blood_type'] = blood_type
    await query.message.reply_text('Please enter the location where you need blood (district, address, or Google Maps link).')
    return LOCATION_FIND

async def show_profile(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT * FROM donors WHERE user_id = ?", (user_id,)) as cursor:
                donor = await cursor.fetchone()

        if donor:
            profile_text = "📋 Your Donor Profile:\n\n"
            profile_text += f"👤 Name: {donor[2]}\n"
            profile_text += f"🩸 Blood Type: {donor[3]}\n"
            profile_text += f"📞 Contact: {donor[6]}\n"
            profile_text += f"📅 Last Donation: {donor[7] if donor[7] else 'Never'}\n"
            
            # Check if 'available' field exists in the donor tuple
            if len(donor) > 8:
                profile_text += f"✅ Available: {'Yes' if donor[8] else 'No'}\n"
            else:
                profile_text += "✅ Available: Yes (Default)\n"

            keyboard = [
                [InlineKeyboardButton("Update Profile", callback_data='update_profile')],
                [InlineKeyboardButton("Back to Main Menu", callback_data='back_to_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if update.callback_query:
                await update.callback_query.message.reply_text(profile_text, reply_markup=reply_markup)
            else:
                await update.message.reply_text(profile_text, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text("You haven't registered as a donor yet.")

    except Exception as e:
        logger.error(f"Database error in show_profile: {e}")
        await update.callback_query.message.reply_text("An error occurred while retrieving your profile. Please try again later.")

    return PROFILE_VIEW

async def back_to_menu(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    keyboard = [
        [InlineKeyboardButton("Donate Blood 🩸", callback_data='donate'),
        InlineKeyboardButton("Find Blood 🔍", callback_data='find')],
        [InlineKeyboardButton("My Profile 👤", callback_data='profile')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.message.edit_text(
        'Welcome back to the main menu! What would you like to do?',
        reply_markup=reply_markup
    )
    return MENU

async def menu_command(update: Update, context: CallbackContext) -> int:
    # Create the menu keyboard
    keyboard = [
        [InlineKeyboardButton("Donate Blood 🩸", callback_data='donate'),
        InlineKeyboardButton("Find Blood 🔍", callback_data='find')],
        [InlineKeyboardButton("My Profile 👤", callback_data='profile')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send a new message with the menu options
    await update.message.reply_text(
        'Welcome back to the main menu! What would you like to do?',
        reply_markup=reply_markup
    )

    return MENU

async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END

async def error_handler(update: Update, context: CallbackContext) -> None:
    logger.warning('Update "%s" caused error "%s"', update, context.error)
    if update.effective_message:
        await update.effective_message.reply_text("An error occurred. Our team has been notified.")

async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        '''
        🩸🇧🇩 Welcome to BloodHeroes Bangladesh! 🦸‍♀️🦸‍♂️

Ready to become a real-life superhero? Your adventure in saving lives starts here! 🚀

🌟 Superhero Command Center:
/start - Activate your hero powers! 💪
/menu - View your superhero options 📋
/help - Call for backup (show this message) 📞
/cancel - Abort mission (cancel current operation) 🚫

🦸 Choose Your Heroic Path:
• Donate Blood 🩸 - Be the hero someone is waiting for!
• Find Blood 🔍 - Lead the search for lifesaving matches!
• Hero Profile 👤 - Power up your alter ego! (for registered heroes)

Remember: Every drop counts, every hero matters! 💖

Stuck on your mission? Don't worry! Your trusty sidekick (that's me!) is here to help. Just ask! 😊

Type /menu to see all your superhero options! Let's save lives together! 🌈✨

Made with ❤️ by 💻 Ahammad Nafiz. Check out more cool projects at github.com/ahammadnafiz!

#BloodHeroesBangladesh #EveryDropCounts #BeAHero
        '''
    )
    await update.message.reply_text(help_text)

def main() -> None:
    # Set up the database before running the bot
    import asyncio
    asyncio.get_event_loop().run_until_complete(setup_database())

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # Add conversation handler with the updated states
    conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start), CommandHandler('menu', menu_command)],
    states={
        MENU: [CallbackQueryHandler(menu_callback)],
        LOCATION: [MessageHandler(filters.LOCATION | filters.TEXT & ~filters.COMMAND, location)],
        BLOOD_TYPE: [CallbackQueryHandler(blood_type_callback)],
        CONTACT: [MessageHandler(filters.TEXT & ~filters.COMMAND, contact)],
        UPDATE_PROFILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_profile)],
        FIND: [CallbackQueryHandler(blood_type_find_callback)],
        LOCATION_FIND: [MessageHandler(filters.LOCATION | filters.TEXT & ~filters.COMMAND, handle_find_blood_location)],
        PROFILE_VIEW: [CallbackQueryHandler(menu_callback)],
        UPDATE_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_name)],
        UPDATE_BLOOD_TYPE: [CallbackQueryHandler(update_blood_type_callback)],
        UPDATE_CONTACT: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_contact)],
        UPDATE_LAST_DONATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_last_donation)],
        UPDATE_LOCATION: [MessageHandler(filters.LOCATION | filters.TEXT & ~filters.COMMAND, update_location)],
        UPDATE_AVAILABILITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_availability)],
    },
    fallbacks=[CommandHandler('cancel', cancel), CommandHandler('menu', menu_command)],
)

    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(paginate_results, pattern=r'^next_page_'))

    # Add standalone command handlers
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("menu", menu_command))

    # Add error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()