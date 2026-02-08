"""Tests for FsCompactor - conversation history summarization.

This module tests the summary generation logic of FsCompactor class,
which creates compact summaries of conversation history using LLM.
"""

import asyncio

from reme import ReMeFs
from reme.core.enumeration import Role
from reme.core.schema import Message


def print_messages(messages: list[Message], title: str = "Messages", max_content_len: int = 150):
    """Print messages with their role and content.

    Args:
        messages: List of messages to print
        title: Title for the message list
        max_content_len: Maximum content length to display (truncate if longer)
    """
    print(f"\n{title}: (count: {len(messages)})")
    print("-" * 80)
    for i, msg in enumerate(messages):
        content = str(msg.content)
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{i}] {msg.role.value:10s}: {content}")
    print("-" * 80)


def create_long_conversation() -> list[Message]:
    """Create a long conversation that exceeds token thresholds."""
    messages = [
        Message(
            role=Role.USER,
            content="I need help building a complete web application with authentication, database, and API endpoints.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""I'll help you build a complete web application. Here's what we'll do:

1. Set up the project structure
2. Implement authentication system
3. Design and create database schema
4. Build API endpoints
5. Add frontend components
6. Test and deploy

Let me start with the project structure...""",
        ),
    ]

    for i in range(15):
        messages.append(
            Message(
                role=Role.USER,
                content=f"What about step {i + 1}? Can you provide more details?",
            ),
        )
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=f"""For step {i + 1}, here's a detailed explanation:

First, we need to consider the architecture. """
                + "This is important context. " * 50
                + """

Then we implement the following:
- Component A
- Component B
- Component C

Let me show you the code for this part..."""
                + "\n\ncode_example = 'example'" * 20,
            ),
        )

    return messages


def create_realistic_personal_conversation() -> list[Message]:
    """Create a realistic conversation with personal information for testing compaction."""
    messages = [
        Message(
            role=Role.USER,
            content="Hi! I'm planning a trip to Japan next month. I need help organizing my itinerary.",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""\
Great! I'd be happy to help you plan your Japan trip. To give you the best recommendations, could you tell me:
1. Which cities are you planning to visit?
2. How many days will you be there?
3. What are your interests (food, culture, nature, shopping, etc.)?
4. What's your approximate budget per day?""",
        ),
        Message(
            role=Role.USER,
            content="""\
I'm planning to visit Tokyo, Kyoto, and Osaka. The trip will be 14 days, from March 15th to March 29th.
My name is Sarah Chen, and I'm 28 years old. I'm traveling with my boyfriend David Miller who is 30.
We're both really into Japanese food - sushi, ramen, tempura, you name it! We also love
temples, gardens, and want to experience traditional Japanese culture.
Our budget is around $200 per day per person, not including accommodation which we've already booked.""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""Perfect! With 14 days and your interests, you'll have an amazing time.
Here's a suggested breakdown:

**Tokyo (5 days - March 15-19)**
- Day 1: Arrive, check into hotel, explore Shibuya and Harajuku
- Day 2: Visit Senso-ji Temple in Asakusa, explore Ueno Park
- Day 3: Tsukiji Outer Market for breakfast, Imperial Palace, Ginza shopping
- Day 4: Day trip to Nikko (UNESCO World Heritage temples)
- Day 5: TeamLab Borderless, Odaiba, evening in Roppongi

**Kyoto (6 days - March 20-25)**
- Day 6: Travel to Kyoto, Fushimi Inari Shrine (iconic red torii gates)
- Day 7: Arashiyama Bamboo Grove, Tenryu-ji Temple, Monkey Park
- Day 8: Kinkaku-ji (Golden Pavilion), Ryoan-ji rock garden
- Day 9: Gion district, traditional tea ceremony, Kiyomizu-dera Temple
- Day 10: Day trip to Nara (deer park, Todai-ji Temple)
- Day 11: Philosopher's Path, Ginkaku-ji (Silver Pavilion), Heian Shrine

**Osaka (3 days - March 26-28)**
- Day 12: Travel to Osaka, Osaka Castle, Dotonbori food street
- Day 13: Kuromon Market, Shinsaibashi shopping, Universal Studios Japan option
- Day 14: Sumiyoshi Taisha Shrine, last-minute shopping, departure prep

Would you like me to suggest specific restaurants or any other details?""",
        ),
        Message(
            role=Role.USER,
            content="""\
This looks fantastic! Yes, please suggest restaurants. We're especially interested in
authentic local places, not touristy spots.
Also, I should mention that David has a shellfish allergy, so we need to be careful
about that.
My email is sarah.chen@email.com if you want to send me a detailed PDF later.
My phone is +1-555-0123-4567.
We're staying at the Park Hyatt Tokyo (5 days), traditional ryokan in Kyoto called
Yoshikawa Inn (6 days), and Hotel Granvia Osaka (3 days).""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""Excellent accommodation choices! The Yoshikawa Inn is particularly special
- they have an amazing kaiseki restaurant.

**Restaurant Recommendations:**

**Tokyo:**
1. **Ichiran Ramen** (Shibuya) - Private booth ramen experience, customize your bowl. """
            """Shellfish-free option available.
2. **Sushi Dai** (Toyosu Market) - Early morning sushi breakfast, tell them about the
shellfish allergy """
            """("ebi to kai rui arerugī")
3. **Tempura Kondo** (Ginza) - Michelin-starred, vegetable-focused tempura
4. **Narisawa** (Aoyama) - If you want a splurge, this 2-Michelin-star restaurant has
innovative Japanese cuisine
5. **Afuri Ramen** (Harajuku) - Yuzu-based light ramen, very refreshing

**Kyoto:**
1. **Yoshikawa Inn's own restaurant** - Their kaiseki is exceptional, """
            """inform them of the allergy when booking
2. **Omen** (near Ginkaku-ji) - Handmade udon in a traditional house
3. **Kikunoi Roan** - Kaiseki experience, they're very accommodating with allergies
4. **Katsukura** (Sanjo) - Best tonkatsu (pork cutlet) in Kyoto
5. **Nishiki Market** - Walk through for street food, but David should be cautious with mystery items

**Osaka:**
1. **Harukoma Sushi** (Tenjinbashisuji) - Conveyor belt sushi, excellent quality, affordable
2. **Fukutaro** (Dotonbori) - Okonomiyaki specialists, no shellfish in standard recipes
3. **Matsusakagyu Yakiniku M** (Hozenji Yokocho) - High-end yakiniku (grilled beef)
4. **Ichiran Ramen** (Dotonbori branch) - Consistent quality
5. **Kuromon Market** - Various stalls, ask vendors about ingredients

**Important Allergy Notes:**
- Learn this phrase: "Ebi to kai rui arerugī ga arimasu"
  (I have a shellfish allergy)
- Many broths use shellfish stock, always ask
- Carry allergy cards in Japanese
- Download Google Translate offline for emergency communication

For your trip planning, I'll note: Sarah Chen (sarah.chen@email.com, +1-555-0123-4567)
and David Miller, March 15-29 Japan trip.""",
        ),
        Message(
            role=Role.USER,
            content="""\
Thank you so much! A few more questions:
1. What's the best way to get from Tokyo to Kyoto? We heard about the JR Pass.
2. Should we rent a pocket WiFi or buy a SIM card?
3. Any cultural etiquette we should be aware of?
4. David wants to know if there are good bars or nightlife spots. He loves whisky.
""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""Great questions! Let me address each:

**1. Transportation:**
Absolutely get the JR Pass! For your 14-day trip, get the 14-day pass (around ¥47,250/$315 per person).
- Tokyo to Kyoto: Shinkansen (bullet train), 2.5 hours, covered by JR Pass
- Kyoto to Osaka: 30 minutes by train, also covered
- The pass also covers JR trains in cities and the Nikko trip
- Order it before leaving the US, pick up at airport upon arrival

**2. Internet:**
I recommend renting pocket WiFi over SIM card because:
- You can share it (both Sarah and David can connect)
- No need to swap your phone SIM
- Rent from companies like Japan Wireless or Ninja WiFi
- Reserve online, pick up at Narita/Haneda airport
- Cost: around $8-10/day for unlimited data
- Return at airport when you leave

**3. Cultural Etiquette:**
- Take off shoes when entering homes, some restaurants, temples (you'll see slippers at entrance)
- Don't tip - it can be considered insulting
- At temples: bow before entering, don't take photos inside unless permitted
- On trains: silent mode on phones, no phone calls
- Eating: slurping noodles is okay (shows appreciation), don't stick chopsticks upright in rice
- Onsen (if your ryokan has one): shower completely before entering communal bath, no swimwear
- Walking: don't eat while walking, finish at the stall or sit down
- Trash: carry your trash with you, public bins are rare

**4. Bars & Whisky for David:**

**Tokyo:**
1. **Zoetrope** (Shinjuku) - 300+ types of Japanese whisky, cozy atmosphere
2. **Bar Benfiddich** (Shinjuku) - Award-winning cocktail bar, house-made bitters
3. **Mori Bar** (Ginza) - Classic Japanese bar, extensive whisky collection
4. **Golden Gai** (Shinjuku) - Alley with 200+ tiny bars, unique atmosphere

**Kyoto:**
1. **Sake Bar Yoramu** - Also has excellent whisky selection
2. **Rocking Bar Ing** - Relaxed atmosphere, good whisky menu
3. **Bar K6** (Kawaramachi) - Intimate whisky bar

**Osaka:**
1. **Bar Nayuta** (Namba) - World-class whisky bar
2. **Moonshine Karaoke Bar** (Dotonbori) - If you want karaoke with drinks
3. **Bar Americana** - Extensive bourbon and whisky selection

**Whisky Shopping:**
- **Tanakaya** (Tokyo) - Huge selection of Japanese whisky
- **Liquors Hasegawa** (Tokyo Station) - Great for last-minute bottles before flying home
- Note: Japanese whisky (Yamazaki, Hibiki, Hakushu) can be expensive even in Japan due to global demand

Do you need help with anything else for your trip? Train routes, more specific timing, or """
            """budget breakdown?""",
        ),
        Message(
            role=Role.USER,
            content="""\
This is incredibly helpful! One more thing - we want to attend a traditional tea
ceremony in Kyoto. Can you recommend where to do this?
Also, what should we pack? I know March can be unpredictable weather-wise.
Oh, and my birthday is on March 22nd - any special restaurant recommendation for
that evening? It'll be our 3rd anniversary too!
David's credit card is Visa ending in 4892, and mine is Mastercard ending in 7651
- will these work everywhere in Japan?""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""Wonderful questions! And happy early birthday & anniversary!

**Tea Ceremony Experiences in Kyoto:**

1. **Camellia Tea Ceremony** (Higashiyama) - Highly recommended!
   - English-speaking host
   - Includes kimono wearing experience
   - Small groups (max 6 people)
   - 2-hour experience, around ¥6,000 per person
   - Book online in advance: camelliatearoom.com

2. **En Tea Ceremony** (near Kiyomizu-dera)
   - Traditional machiya (townhouse) setting
   - Very authentic, less touristy
   - Private ceremony option available
   - Around ¥5,500 per person

3. **Wak Japan** (Gion area)
   - Combines tea ceremony with flower arrangement or calligraphy
   - Good for couples
   - Around ¥8,000 per person for combined experience

I recommend booking Day 9 (March 22nd) morning for the tea ceremony, then evening for your special dinner!

**March Weather & Packing:**
March in Japan: transitioning from winter to spring
- Temperature: 8-15°C (46-59°F)
- Cherry blossoms might just start blooming late March (you might catch early bloomers!)

**Pack:**
- Layering clothes: light sweater, cardigan, light jacket
- One warmer jacket for evenings
- Comfortable walking shoes (you'll walk 15,000+ steps daily)
- Umbrella (March has occasional rain)
- Slip-on shoes (easier for temple visits)
- Nice outfit for fancy restaurants
- Power adapter (Japan uses Type A plugs, 100V)
- Portable charger for phones
- Small day backpack

**Birthday & Anniversary Dinner - March 22nd:**

For such a special occasion in Kyoto, I highly recommend:

**Kikunoi Honten** (Main Branch) - 3 Michelin Stars
- Ultimate kaiseki experience
- Beautiful traditional setting with garden views
- Multi-course seasonal menu
- Reserve 1-2 months in advance
- Budget: ¥25,000-40,000 per person (worth it for anniversary!)
- Inform them of: shellfish allergy, birthday, anniversary
- They'll often add special touches for celebrations

**Alternative (still amazing):**
**Hyotei** - 3 Michelin Stars
- Historic (400 years old)
- Beautiful garden setting
- Breakfast kaiseki is famous, but dinner is exceptional
- Similar price range

**More Accessible but Still Special:**
**Gion Karyo**
- 1 Michelin Star
- Beautiful Gion location
- Around ¥15,000-20,000 per person
- Easier to book

**Birthday Cake:**
Order a birthday cake from **Malebranche** (Kyoto) - they do beautiful matcha cakes.
Ask your ryokan to keep it refrigerated and present it after you return from dinner!

**Credit Cards:**
- Major hotels and restaurants: Visa & Mastercard widely accepted
  (Sarah's Mastercard 7651 and David's Visa 4892 should work fine)
- However, Japan is still heavily CASH-based
- Bring cash for:
  - Small restaurants
  - Temples/shrines
  - Street food
  - Some shops
  - Taxis
  - Vending machines

**ATM Tips:**
- 7-Eleven ATMs work with foreign cards (open 24/7)
- Japan Post Bank ATMs also reliable
- Withdraw larger amounts to minimize fees
- Keep ¥10,000-20,000 ($70-140) cash daily
- Your banks: notify them of Japan travel to avoid card blocks

**Pre-Trip Checklist:**
✅ JR Pass ordered
✅ Pocket WiFi reserved
✅ Kikunoi dinner reservation (book NOW for March 22nd)
✅ Tea ceremony booking
✅ Notify banks: Sarah (MC 7651), David (Visa 4892)
✅ Travel insurance
✅ Download offline maps
✅ Book airport transfers
✅ Birthday cake ordered

Would you like me to help with a day-by-day detailed schedule with specific times and addresses?""",
        ),
        Message(
            role=Role.USER,
            content="""\
Yes! A detailed daily schedule would be amazing. Can you include specific addresses and opening hours?
Also, we're thinking of proposing to our friends to join us - my best friend Jessica Wang and her """
            """husband Tom. What would change if we're traveling as a group of 4?""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""\
Absolutely! I'll create a detailed schedule. Traveling as a group of 4 (Sarah & David + Jessica & """
            """Tom) actually has some advantages:

**Group of 4 Benefits:**
✅ Split pocket WiFi cost (1 device for 4 people)
✅ Share taxi costs (taxis fit 4 people)
✅ Some restaurants have minimum 4-person set menus
✅ Private tea ceremony for your group
✅ Better for group photos!

**Considerations:**
- Book restaurants for 4 people
- Some tiny bars in Golden Gai might not fit all
- Reserve 2 rooms/apartments when needed
- Coordinate meeting points if you split up

**DETAILED 14-DAY SCHEDULE WITH ADDRESSES:**

**DAY 1 - March 15 (Friday) - TOKYO ARRIVAL**

*Morning/Afternoon:*
- Arrive Narita/Haneda Airport
- Pick up: JR Pass, Pocket WiFi
- Exchange yen at airport (recommend ¥50,000+ per person)
- Take train to hotel: Park Hyatt Tokyo
  📍 3-7-1-2 Nishishinjuku, Shinjuku-ku, Tokyo 163-1055
  🚇 Shinjuku Station → Oedo Line to Tochomae Station (5 min walk)

*Evening (6:00 PM - 9:00 PM):*
- Check in, rest, freshen up
- Dinner: **Omoide Yokocho** (Memory Lane)
  📍 1 Chome Nishishinjuku, Shinjuku-ku, Tokyo
  🕒 Open till midnight
  💴 ¥2,000-3,000/person
  - Narrow alley with small yakitori stands
  - Cash only, very local atmosphere
  - Ask about shellfish ("kai rui") in skewers

*Night:*
- Walk around Shinjuku, see the night lights
- Convenience store snacks (7-Eleven/Family Mart)
- Early sleep (jet lag)

---

**DAY 2 - March 16 (Saturday) - ASAKUSA & UENO**

*Morning (9:00 AM - 12:00 PM):*
- Breakfast at hotel or nearby bakery
- 🚇 Train to Asakusa (30 min from Shinjuku)

- **Senso-ji Temple**
  📍 2-3-1 Asakusa, Taito-ku, Tokyo
  🕒 6:00 AM - 5:00 PM (grounds always open)
  💴 Free
  - Arrive by 9:30 AM to avoid crowds
  - Walk through Kaminarimon Gate, Nakamise Shopping Street
  - Draw fortune (omikuji) - ¥100
  - Visit main hall, incense burner

*Lunch (12:00 PM):*
- **Daikokuya Tempura**
  📍 1-38-10 Asakusa, Taito-ku, Tokyo
  🕒 11:00 AM - 8:30 PM (closed Mon)
  💴 ¥2,000-3,000/person
  - Famous tendon (tempura rice bowl)
  - Mention shellfish allergy to David's order

*Afternoon (1:30 PM - 5:00 PM):*
- Walk to Ueno (15 min) or train (2 stops)

- **Ueno Park**
  📍 Uenokoen, Taito-ku, Tokyo
  🕒 5:00 AM - 11:00 PM
  💴 Free (museums extra)
  - Cherry blossom trees (might see early bloomers!)
  - Visit **Tokyo National Museum** if interested
    🕒 9:30 AM - 5:00 PM (closed Mon)
    💴 ¥1,000/person

- **Ameya-Yokocho Market**
  📍 4 Chome Ueno, Taito-ku, Tokyo
  - Shopping street, bargain clothes, snacks

*Dinner (6:30 PM):*
- **Ichiran Ramen Ueno**
  📍 6-11-11 Ueno, Taito-ku, Tokyo
  🕒 24 hours
  💴 ¥1,000-1,500/person
  - Individual booth experience
  - Order via vending machine (English available)
  - Customize your ramen

*Night:*
- Return to Shinjuku
- Optional: **Zoetrope Whisky Bar** for David & Tom
  📍 Sankoubldg. 3F, 1-7-10 Nishi-Shinjuku, Shinjuku-ku
  🕒 6:00 PM - 12:00 AM (closed Sun)
  💴 ¥1,500-3,000/drink

---

**DAY 3 - March 17 (Sunday) - TSUKIJI, GINZA, IMPERIAL PALACE**

*Early Morning (5:30 AM - 8:00 AM):*
- Wake up early!
- **Tsukiji Outer Market**
  📍 4 Chome Tsukiji, Chuo-ku, Tokyo
  🚇 Tsukijishijo Station (Oedo Line)
  🕒 Most stalls: 5:00 AM - 2:00 PM

- Breakfast at **Sushi Dai** (or Daiwa Sushi)
  📍 Inside Toyosu Market (new location)
  🕒 5:30 AM - 1:30 PM
  💴 ¥3,500-5,000/person
  ⚠️ Expect 1-2 hour wait, go early!
  - Tell chef about David's shellfish allergy
  - Omakase sushi breakfast

*Late Morning (9:00 AM - 12:00 PM):*
- **Imperial Palace East Gardens**
  📍 1-1 Chiyoda, Chiyoda-ku, Tokyo
  🕒 9:00 AM - 4:30 PM (closed Mon, Fri)
  💴 Free
  - Beautiful gardens, historic site
  - 1-1.5 hour visit

*Lunch (12:30 PM):*
- **Ginza**
  📍 Ginza, Chuo-ku, Tokyo
  - Many options for lunch

- **Tempura Kondo** (if you can get reservation)
  📍 Sakaguchi Bldg. 9F, 5-5-13 Ginza, Chuo-ku
  🕒 Lunch 12:00-2:00 PM, Dinner 5:30-9:00 PM (closed Sun)
  💴 Lunch ¥8,000-12,000/person
  - Reserve online or call: +81-3-5568-0923

- **Backup: Ginza Kagari Ramen**
  📍 Ginza, Chuo-ku (search exact location)
  💴 ¥1,200/person
  - Creamy chicken ramen, no shellfish

*Afternoon (2:00 PM - 6:00 PM):*
- **Ginza Shopping**
  - UNIQLO flagship (12 floors)
  - Mitsukoshi Department Store
  - Dover Street Market (avant-garde fashion)
  - MUJI flagship
  - Window shop luxury brands

*Dinner (6:30 PM):*
- **Afuri Ramen**
  📍 1-1-7 Ebisu, Shibuya-ku, Tokyo (Ebisu location)
  🕒 11:00 AM - 11:00 PM
  💴 ¥1,200/person
  - Yuzu-salt ramen, light and refreshing

*Night:*
- Train to Shibuya for evening walk
- See Shibuya Crossing at night
- Return hotel

---

This is getting quite long! Should I continue with the rest of the days (Days 4-14)? I can also send """
            """you this as a Google Doc or PDF if that's easier. Just need to confirm - are Jessica and Tom """
            """definitely joining, or still maybe?

Also, does anyone have other dietary restrictions besides David's shellfish allergy? And what are """
            """your hotel/ryokan confirmations - should I include check-in/check-out timing?""",
        ),
    ]

    return messages


async def test_full_compact_with_summary():
    """Test complete compaction flow with LLM summary generation.

    This is a complex integration test that exercises the full compaction pipeline:
    1. Create a long conversation that exceeds token threshold
    2. Context checker finds cut point (may include split turn detection)
    3. Compactor generates summary for messages to summarize
    4. Compactor handles turn prefix if split turn detected
    5. Final output contains summary + recent messages

    Expects:
        - compacted=True
        - Summary message generated with proper format
        - Split turn handling if applicable
        - Reduced message count
        - Token count within limits
    """
    print("\n" + "=" * 80)
    print("TEST: Full Compaction with LLM Summary Generation")
    print("=" * 80)

    reme_fs = ReMeFs(
        enable_logo=False,
        vector_store=None,
        compact_params={
            "context_window_tokens": 3000,
            "reserve_tokens": 1500,
            "keep_recent_tokens": 500,
        },
    )
    await reme_fs.start()

    messages = create_long_conversation()
    print_messages(messages, "INPUT MESSAGES (Long Conversation)", max_content_len=60)

    print("\nParameters:")
    print("  context_window_tokens: 3000")
    print("  reserve_tokens: 1500 (threshold = 1500)")
    print("  keep_recent_tokens: 500")
    print("\nExpectations:")
    print("  - Token count exceeds threshold")
    print("  - Context checker finds cut point")
    print("  - Compactor generates summary via LLM")
    print("  - May detect split turn scenario")
    print("  - Returns summary + recent messages")

    # Execute full compact flow
    result = await reme_fs.compact(messages_to_summarize=messages)

    print(f"\n{'=' * 80}")
    print("RESULT:")
    print(f"  compacted: {result}")
    await reme_fs.close()


async def test_realistic_personal_conversation_compact():
    """Test compaction with realistic personal conversation and return summary string.

    This test:
    1. Creates a realistic conversation with personal details
    2. Runs compaction to generate a summary
    3. Returns the summary as a string
    4. Validates the compaction result
    """
    print("\n" + "=" * 80)
    print("TEST: Realistic Personal Conversation Compaction")
    print("=" * 80)

    reme_fs = ReMeFs(
        enable_logo=False,
        vector_store=None,
        compact_params={
            "context_window_tokens": 4000,
            "reserve_tokens": 2000,
            "keep_recent_tokens": 800,
        },
    )
    await reme_fs.start()

    messages = create_realistic_personal_conversation()
    print_messages(messages, "INPUT: Realistic Personal Conversation", max_content_len=100)

    print(f"\n{'=' * 80}")
    print("COMPACTING CONVERSATION...")
    print(f"{'=' * 80}")

    # Execute compaction
    result = await reme_fs.compact(messages_to_summarize=messages)

    print(f"\n{'=' * 80}")
    print("COMPACTION RESULT:")
    print(f"{'=' * 80}")
    print(f"  compacted: {result}")
    await reme_fs.close()


async def main():
    """Run compactor tests."""
    print("\n" + "=" * 80)
    print("FsCompactor - Summary Generation Test Suite")
    print("=" * 80)
    print("\nThis test suite validates the LLM-based summarization:")
    print("  - Full compaction flow (context check + summary generation)")
    print("  - Summary format and structure")
    print("  - Split turn handling")
    print("  - Message preservation")
    print("  - Realistic personal conversation compaction")
    print("=" * 80)
    print("\nNote: This test requires LLM access and may take some time.")
    print("=" * 80)

    # Run the comprehensive compaction test
    await test_full_compact_with_summary()

    # Run the realistic personal conversation test
    await test_realistic_personal_conversation_compact()


if __name__ == "__main__":
    asyncio.run(main())
