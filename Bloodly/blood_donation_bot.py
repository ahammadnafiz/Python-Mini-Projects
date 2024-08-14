import os
import logging
from datetime import datetime
import re
from typing import List, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
MENU, LOCATION, BLOOD_TYPE, CONTACT, PROFILE, FIND, EMERGENCY, LOCATION_FIND = range(8)
BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Initialize the geolocator
geolocator = Nominatim(user_agent="blood_donation_bot")

# Database setup
DB_PATH = 'blood_donation.db'

async def setup_database():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
            CREATE TABLE IF NOT EXISTS donors (
                id INTEGER PRIMARY KEY,
                user_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                blood_type TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                contact TEXT NOT NULL,
                last_donation TEXT
            )
            """)
            await db.commit()
    except Exception as e:
        logger.error(f"Database setup error: {e}")

def parse_dms_coordinate(coord_str):
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NS])\s*(\d+)°(\d+)'([\d.]+)\"([EW])"
    match = re.match(pattern, coord_str)
    if not match:
        return None
    
    lat_d, lat_m, lat_s, lat_dir, lon_d, lon_m, lon_s, lon_dir = match.groups()
    
    lat = (float(lat_d) + float(lat_m)/60 + float(lat_s)/3600) * (1 if lat_dir == 'N' else -1)
    lon = (float(lon_d) + float(lon_m)/60 + float(lon_s)/3600) * (1 if lon_dir == 'E' else -1)
    
    return lat, lon

async def start(update: Update, context: CallbackContext) -> int:
    user_id = update.effective_user.id
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT * FROM donors WHERE user_id = ?", (user_id,))
        user_is_registered = await cursor.fetchone() is not None

    keyboard = [
        [InlineKeyboardButton("Donate Blood 🩸", callback_data='donate'),
         InlineKeyboardButton("Find Blood 🔍", callback_data='find')],
        [InlineKeyboardButton("Emergency Request 🚨", callback_data='emergency')]
    ]
    if user_is_registered:
        keyboard.append([InlineKeyboardButton("My Profile 👤", callback_data='profile')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        'Welcome to the Bangladesh Blood Donation Bot! 🇧🇩\n\n'
        'This bot helps connect blood donors with those in need. '
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
    elif query.data == 'emergency':
        keyboard = [[InlineKeyboardButton(bt, callback_data=f'blood_{bt}') for bt in BLOOD_TYPES[i:i+2]] for i in range(0, len(BLOOD_TYPES), 2)]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.message.reply_text('What blood type do you need?', reply_markup=reply_markup)
        return EMERGENCY
    elif query.data == 'profile':
        return await show_profile(update, context)
    else:
        await query.message.reply_text('Invalid choice. Please select from the options provided.')
        return MENU

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
            (user_id, name, blood_type, latitude, longitude, contact, last_donation) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, name, context.user_data['blood_type'], 
                  context.user_data['latitude'], context.user_data['longitude'], 
                  context.user_data['contact'], last_donation))
            await db.commit()
        
        await update.message.reply_text('Thank you for registering as a donor! 🎉\n\nYour information has been saved successfully.')
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.message.reply_text('An error occurred while saving your information. Please try again later.')
    
    return ConversationHandler.END

async def find_nearest_donors(lat: float, lon: float, blood_type: str, limit: int = 5) -> List[Tuple]:
    logger.info(f"Finding nearest donors to location: ({lat}, {lon}) for blood type: {blood_type}")

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # First, check if there are any donors with the exact blood type
            query = "SELECT * FROM donors WHERE blood_type = ?"
            params = [blood_type]
            
            async with db.execute(query, params) as cursor:
                donors = await cursor.fetchall()

            if not donors:
                # If no exact matches, find compatible blood types
                compatible_types = get_compatible_blood_types(blood_type)
                query = f"SELECT * FROM donors WHERE blood_type IN ({','.join(['?']*len(compatible_types))})"
                params = compatible_types
                
                async with db.execute(query, params) as cursor:
                    donors = await cursor.fetchall()

            df = pd.DataFrame(donors, columns=['id', 'user_id', 'name', 'blood_type', 'latitude', 'longitude', 'contact', 'last_donation'])

            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])

            df['distance'] = df.apply(lambda row: geodesic((lat, lon), (row['latitude'], row['longitude'])).km, axis=1)
            df_sorted = df.sort_values(by='distance')
            nearest_donors = df_sorted.head(limit)

            return [(row['blood_type'], row['distance'], row['contact'], row['latitude'], row['longitude']) for _, row in nearest_donors.iterrows()]

    except Exception as e:
        logger.error(f"Error while finding nearest donors: {e}")
        return []

def get_compatible_blood_types(blood_type: str) -> List[str]:
    compatibility = {
        'O-': ['O-'],
        'O+': ['O-', 'O+'],
        'A-': ['O-', 'A-'],
        'A+': ['O-', 'O+', 'A-', 'A+'],
        'B-': ['O-', 'B-'],
        'B+': ['O-', 'O+', 'B-', 'B+'],
        'AB-': ['O-', 'A-', 'B-', 'AB-'],
        'AB+': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+']
    }
    return compatibility.get(blood_type.upper(), [])

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
    nearest_donors = await find_nearest_donors(latitude, longitude, blood_type=needed_blood_type, limit=5)
    
    if not nearest_donors:
        await update.message.reply_text(f"No donors with blood type {needed_blood_type} or compatible types found in the area. Please try a wider search area.")
    else:
        response = f"Nearest donors for blood type {needed_blood_type}:\n\n"
        for i, (blood_type, distance, contact, donor_lat, donor_lon) in enumerate(nearest_donors, 1):
            response += f"{i}. Blood Type: {blood_type}\n"
            response += f"   Distance: {distance:.2f} km\n"
            response += f"   Contact: {contact}\n"
            response += f"   Location: https://www.google.com/maps?q={donor_lat},{donor_lon}\n\n"

        await update.message.reply_text(response)
    
    return ConversationHandler.END

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
            profile_text = f"📋 Your Donor Profile:\n\n"
            profile_text += f"👤 Name: {donor[2]}\n"
            profile_text += f"🩸 Blood Type: {donor[3]}\n"
            profile_text += f"📞 Contact: {donor[6]}\n"
            profile_text += f"📅 Last Donation: {donor[7] if donor[7] else 'Never'}\n"
            
            keyboard = [[InlineKeyboardButton("Update Profile", callback_data='update_profile')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.callback_query.message.reply_text(profile_text, reply_markup=reply_markup)
        else:
            await update.callback_query.message.reply_text("You haven't registered as a donor yet.")
    
    except Exception as e:
        logger.error(f"Database error: {e}")
        await update.callback_query.message.reply_text("An error occurred while retrieving your profile. Please try again later.")
    
    return ConversationHandler.END

async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END

async def error_handler(update: Update, context: CallbackContext) -> None:
    logger.warning('Update "%s" caused error "%s"', update, context.error)
    if update.effective_message:
        await update.effective_message.reply_text("An error occurred. Our team has been notified.")

async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "Welcome to the Bangladesh Blood Donation Bot! 🇧🇩🩸\n\n"
        "Here are the available commands:\n\n"
        "/start - Start the bot and see the main menu\n"
        "/help - Show this help message\n"
        "/cancel - Cancel the current operation\n\n"
        "Use the buttons to:\n"
        "• Donate Blood 🩸\n"
        "• Find Blood 🔍\n"
        "• Make an Emergency Request 🚨\n"
        "• View or Update Your Profile 👤 (if registered)\n\n"
        "If you need further assistance, please don't hesitate to ask!"
    )
    await update.message.reply_text(help_text)

def main() -> None:
    # Set up the database before running the bot
    import asyncio
    asyncio.get_event_loop().run_until_complete(setup_database())
    
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # Add conversation handler with the states MENU, LOCATION, BLOOD_TYPE, CONTACT, PROFILE, FIND, EMERGENCY, LOCATION_FIND
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MENU: [CallbackQueryHandler(menu_callback)],
            LOCATION: [MessageHandler(filters.LOCATION | filters.TEXT & ~filters.COMMAND, location)],
            BLOOD_TYPE: [CallbackQueryHandler(blood_type_callback)],
            CONTACT: [MessageHandler(filters.TEXT & ~filters.COMMAND, contact)],
            PROFILE: [MessageHandler(filters.TEXT & ~filters.COMMAND, profile)],
            FIND: [CallbackQueryHandler(blood_type_find_callback)],
            LOCATION_FIND: [MessageHandler(filters.LOCATION | filters.TEXT & ~filters.COMMAND, handle_find_blood_location)],
            EMERGENCY: [CallbackQueryHandler(emergency_callback)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)

    # Add standalone command handlers
    application.add_handler(CommandHandler("help", help_command))

    # Add error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()