import pandas as pd
from cellula_app.models import Booking
from datetime import datetime

def run():
    csv_path = 'first inten project.csv'
    data = pd.read_csv(csv_path)

    # Clean column names (remove spaces)
    data.columns = data.columns.str.strip()

    # Convert 'date of reservation' to datetime
    data['date of reservation'] = pd.to_datetime(data['date of reservation'], errors='coerce')

    # Replace NaT (Not a Time) with None for compatibility with Django
    data['date of reservation'] = data['date of reservation'].where(data['date of reservation'].notna(), None)

    # Loop through rows and create Booking objects
    bookings = [
        Booking(
            booking_id=row['Booking_ID'],
            number_of_adults=row['number of adults'],
            number_of_children=row['number of children'],
            number_of_weekend_nights=row['number of weekend nights'],
            number_of_week_nights=row['number of week nights'],
            type_of_meal=row['type of meal'],
            car_parking_space=row['car parking space'],
            room_type=row['room type'],
            lead_time=row['lead time'],
            market_segment_type=row['market segment type'],
            repeated=row['repeated'],
            p_c=row['P-C'],
            p_not_c=row['P-not-C'],
            average_price=row['average price'],
            special_requests=row['special requests'],
            date_of_reservation=row['date of reservation'] if pd.notna(row['date of reservation']) else None,
            booking_status=row['booking status']
        )
        for _, row in data.iterrows()
    ]

    # Bulk insert for performance
    Booking.objects.bulk_create(bookings, batch_size=1000)
    print("Data loaded successfully!")
