# Setup for the heat map generator

from PIL import ImageFont

# str(int(round(reg_value * 100,0))) + "%"
# reg_value_str = str(int(round(reg_value * 100, 0))) + "%"
#
# reg_value = round(row[x], 2)
# reg_value_str = str(int(round(reg_value * 100,0))) + "%"


def draw_fo_outline(draw, fo_idx, font):
    """
    Draw the outline based on the given fo_idx using specific coordinates and fills.

    Parameters:
    - draw: The drawing object to draw shapes on.
    - fo_idx: The index to determine the specific outline to draw.
    - font: The font to use for drawing text on the outline.

    Returns:
    None
    """

    _mapper = gen_invert_geo_mapper()
    _fills = {
        90: '#bab638',
        91: '#34aec9',
        92: '#7ab06d',
        93: '#db7365',
        94: '#ba8fc9',
        95: '#d48d22',
        96: '#3865d6',
        97: '#00ad85',
        98: '#00ad85'
    }

    def draw_line(_draw, _coords, _fill):
        _draw.line(_coords, fill=_fill, width=5, joint='curve')

    if fo_idx == 90:
        draw_line(
            draw,
            _coords=(
                530, 190, 650, 190, 650, 270, 870, 270, 870, 360, 760, 360,
                760, 680, 530, 680, 530, 600, 420, 600, 420, 520, 310, 520,
                310, 350, 420, 350, 420, 270, 530, 270, 530, 190
            ),
            _fill=_fills[fo_idx])
        draw.text((685, 210), _mapper[fo_idx], font=font, fill=_fills[fo_idx])
    
    elif fo_idx == 91:
        draw_line(
            draw,
            _coords=(
                290, 90, 410, 90, 410, 170, 520, 170, 520, 260,
                410, 260, 410, 340, 300, 340, 300, 420, 70, 420,
                70, 330, 180, 330, 180, 170, 290, 170, 290, 90
            ),
            _fill=_fills[fo_idx])
        draw_line(
            draw,
            _coords=(
                1100, 110, 1220, 110, 1220, 280, 990, 280, 990, 190, 1100, 190, 1100, 110
            ),
            _fill=_fills[fo_idx])
        draw.text((960, 130), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 92:
        draw_line(
            draw,
            _coords=(
                770, 370, 880, 370, 880, 290, 1220, 290, 1220, 540, 1110, 540,
                1110, 620, 770, 620, 770, 370
            ),
            _fill=_fills[fo_idx])
        draw.text((1150, 575), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 93:
        draw_line(
            draw,
            _coords=(
                770, 630, 890, 630, 890, 710, 1000, 710, 1000, 800,
                220, 800, 220, 710, 770, 710, 770, 630
            ),
            _fill=_fills[fo_idx])
        draw.text((335, 650), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 94:
        draw_line(
            draw,
            _coords=(
                550, 810, 1000, 810, 1000, 900, 890, 900, 890, 980,
                550, 980, 550, 810
            ),
            _fill=_fills[fo_idx])
        draw.text((400, 863), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 95:
        draw_line(
            draw,
            _coords=(
                1230, 210, 1350, 210, 1350, 290, 1460, 290, 1460, 460, 1350, 460,
                1350, 540, 1230, 540, 1230, 210
            ),
            _fill=_fills[fo_idx])
        draw.text((1380, 230), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 96:
        draw_line(
            draw,
            _coords=(
                1580, 230, 1700, 230, 1700, 310, 1810, 310, 1810, 400, 1700, 400,
                1700, 560, 1590, 560, 1590, 640, 1470, 640, 1470, 560, 1360, 560,
                1360, 470, 1470, 470, 1470, 310, 1580, 310, 1580, 230
            ),
            _fill=_fills[fo_idx])
        draw.text((1620, 590), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 97:
        draw_line(
            draw,
            _coords=(
                1930, 170, 2160, 170, 2160, 260, 2050, 260, 2050, 330, 2160, 330,
                2160, 420, 1940, 420, 1940, 490, 2050, 490, 2050, 580, 1930, 580,
                1930, 500, 1830, 500, 1830, 580, 1710, 580, 1710, 410, 1820, 410,
                1820, 250, 1930, 250, 1930, 170
            ),
            _fill=_fills[fo_idx])
        draw.text((1820, 190), _mapper[fo_idx], font=font, fill=_fills[fo_idx])

    elif fo_idx == 98:
        pass
        # draw_line(
        #     draw,
        #     _coords=(
        #         1930, 170, 2160, 170, 2160, 260, 2050, 260, 2050, 330, 2160, 330,
        #         2160, 420, 1940, 420, 1940, 490, 2050, 490, 2050, 580, 1930, 580,
        #         1930, 500, 1830, 500, 1830, 580, 1710, 580, 1710, 410, 1820, 410,
        #         1820, 250, 1930, 250, 1930, 170
        #     ),
        #     _fill=_fills[fo_idx])
        # draw.text((1820, 190), _mapper[fo_idx], font=font, fill=_fills[fo_idx])


def gen_geo_map():
    """
    Generate a dictionary mapping regions in Russia to their corresponding numerical codes.
    """
    return {
        "Россия": 0,
        "Алтайский край": 1,
        "Амурская область": 2,
        "Архангельская область": 3,
        "Астраханская область": 4,
        "Белгородская область": 5,
        "Брянская область": 6,
        "Владимирская область": 7,
        "Волгоградская область": 8,
        "Вологодская область": 9,
        "Воронежская область": 10,
        "Донецкая Народная Республика": 11,
        "Еврейская АО": 12,
        "Забайкальский край": 13,
        "Запорожская область": 14,
        "Ивановская область": 15,
        "Иркутская область": 16,
        "Кабардино-Балкарская Республика": 17,
        "Калининградская область": 18,
        "Калужская область": 19,
        "Камчатский край": 20,
        "Карачаево-Черкесская Республика": 21,
        "Кемеровская область": 22,
        "Кировская область": 23,
        "Костромская область": 24,
        "Краснодарский край": 25,
        "Красноярский край": 26,
        "Курганская область": 27,
        "Курская область": 28,
        "Ленинградская область": 29,
        "Липецкая область": 30,
        "Луганская Народная Республика": 31,
        "Магаданская область": 32,
        "Москва": 33,
        "Московская область": 34,
        "Мурманская область": 35,
        "Ненецкий АО": 36,
        "Нижегородская область": 37,
        "Новгородская область": 38,
        "Новосибирская область": 39,
        "Омская область": 40,
        "Оренбургская область": 41,
        "Орловская область": 42,
        "Пензенская область": 43,
        "Пермский край": 44,
        "Приморский край": 45,
        "Псковская область": 46,
        "Республика Адыгея": 47,
        "Республика Алтай": 48,
        "Республика Башкортостан": 49,
        "Республика Бурятия": 50,
        "Республика Дагестан": 51,
        "Республика Ингушетия": 52,
        "Республика Калмыкия": 53,
        "Республика Карелия": 54,
        "Республика Коми": 55,
        "Республика Крым": 56,
        "Республика Марий Эл": 57,
        "Республика Мордовия": 58,
        "Республика Саха": 59,
        "Республика Северная Осетия": 60,
        "Республика Татарстан": 61,
        "Республика Тыва": 62,
        "Республика Хакасия": 63,
        "Ростовская область": 64,
        "Рязанская область": 65,
        "Самарская область": 66,
        "Санкт-Петербург": 67,
        "Саратовская область": 68,
        "Сахалинская область": 69,
        "Свердловская область": 70,
        "Севастополь": 71,
        "Смоленская область": 72,
        "Ставропольский край": 73,
        "Тамбовская область": 74,
        "Тверская область": 75,
        "Томская область": 76,
        "Тульская область": 77,
        "Тюменская область": 78,
        "Удмуртская Республика": 79,
        "Ульяновская область": 80,
        "Хабаровский край": 81,
        "Ханты-Мансийский АО": 82,
        "Херсонская область": 83,
        "Челябинская область": 84,
        "Чеченская Республика": 85,
        "Чувашская Республика": 86,
        "Чукотский АО": 87,
        "Ямало-Ненецкий АО": 88,
        "Ярославская область": 89,
        "ЦФО": 90,
        "СЗФО": 91,
        "ПФО": 92,
        "ЮФО": 93,
        "СКФО": 94,
        "УФО": 95,
        "СФО": 96,
        "ДФО": 97,
        "НР": 98,
        "Восточный административный округ": 99,
        "Западный административный округ": 100,
        "Зеленоградский административный округ": 101,
        "Новомосковский административный округ": 102,
        "Северный административный округ": 103,
        "Северо-Восточный административный округ": 104,
        "Северо-Западный административный округ": 105,
        "Троицкий административный округ": 106,
        "Центральный административный округ": 107,
        "Юго-Восточный административный округ": 108,
        "Юго-Западный административный округ": 109,
        "Южный административный округ": 110,
    }


def gen_invert_geo_mapper():
    """
    This function generates and returns a dictionary mapping numeric keys to corresponding geographical locations in Russia.
    """
    return {
         0: 'Россия',
         1: 'Алтайский край',
         2: 'Амурская область',
         3: 'Архангельская область',
         4: 'Астраханская область',
         5: 'Белгородская область',
         6: 'Брянская область',
         7: 'Владимирская область',
         8: 'Волгоградская область',
         9: 'Вологодская область',
         10: 'Воронежская область',
         11: 'Донецкая Народная Республика',
         12: 'Еврейская АО',
         13: 'Забайкальский край',
         14: 'Запорожская область',
         15: 'Ивановская область',
         16: 'Иркутская область',
         17: 'Кабардино-Балкарская Республика',
         18: 'Калининградская область',
         19: 'Калужская область',
         20: 'Камчатский край',
         21: 'Карачаево-Черкесская Республика',
         22: 'Кемеровская область',
         23: 'Кировская область',
         24: 'Костромская область',
         25: 'Краснодарский край',
         26: 'Красноярский край',
         27: 'Курганская область',
         28: 'Курская область',
         29: 'Ленинградская область',
         30: 'Липецкая область',
         31: 'Луганская Народная Республика',
         32: 'Магаданская область',
         33: 'Москва',
         34: 'Московская область',
         35: 'Мурманская область',
         36: 'Ненецкий АО',
         37: 'Нижегородская область',
         38: 'Новгородская область',
         39: 'Новосибирская область',
         40: 'Омская область',
         41: 'Оренбургская область',
         42: 'Орловская область',
         43: 'Пензенская область',
         44: 'Пермский край',
         45: 'Приморский край',
         46: 'Псковская область',
         47: 'Республика Адыгея',
         48: 'Республика Алтай',
         49: 'Республика Башкортостан',
         50: 'Республика Бурятия',
         51: 'Республика Дагестан',
         52: 'Республика Ингушетия',
         53: 'Республика Калмыкия',
         54: 'Республика Карелия',
         55: 'Республика Коми',
         56: 'Республика Крым',
         57: 'Республика Марий Эл',
         58: 'Республика Мордовия',
         59: 'Республика Саха',
         60: 'Республика Северная Осетия',
         61: 'Республика Татарстан',
         62: 'Республика Тыва',
         63: 'Республика Хакасия',
         64: 'Ростовская область',
         65: 'Рязанская область',
         66: 'Самарская область',
         67: 'Санкт-Петербург',
         68: 'Саратовская область',
         69: 'Сахалинская область',
         70: 'Свердловская область',
         71: 'Севастополь',
         72: 'Смоленская область',
         73: 'Ставропольский край',
         74: 'Тамбовская область',
         75: 'Тверская область',
         76: 'Томская область',
         77: 'Тульская область',
         78: 'Тюменская область',
         79: 'Удмуртская Республика',
         80: 'Ульяновская область',
         81: 'Хабаровский край',
         82: 'Ханты-Мансийский АО',
         83: 'Херсонская область',
         84: 'Челябинская область',
         85: 'Чеченская Республика',
         86: 'Чувашская Республика',
         87: 'Чукотский АО',
         88: 'Ямало-Ненецкий АО',
         89: 'Ярославская область',
         90: 'ЦФО',
         91: 'СЗФО',
         92: 'ПФО',
         93: 'ЮФО',
         94: 'СКФО',
         95: 'УФО',
         96: 'СФО',
         97: 'ДФО',
         98: 'НР',
         99: 'Восточный административный округ',
         100: 'Западный административный округ',
         101: 'Зеленоградский административный округ',
         102: 'Новомосковский административный округ',
         103: 'Северный административный округ',
         104: 'Северо-Восточный административный округ',
         105: 'Северо-Западный административный округ',
         106: 'Троицкий административный округ',
         107: 'Центральный административный округ',
         108: 'Юго-Восточный административный округ',
         109: 'Юго-Западный административный округ',
         110: 'Южный административный округ'
    }


def gen_fo_mapper():
    """
    A function that generates a dictionary mapping regions to federal districts.
    Each key corresponds to a specific list of integers.
    Returns the generated dictionary.
    """
    return {
        90: [
            5,
            6,
            7,
            10,
            15,
            19,
            24,
            28,
            30,
            33,
            34,
            42,
            65,
            72,
            74,
            75,
            77,
            89,
        ],
        91: [
            3,
            9,
            18,
            29,
            35,
            36,
            38,
            46,
            54,
            55,
            67,
        ],
        92: [
            23,
            37,
            41,
            43,
            44,
            49,
            57,
            58,
            61,
            66,
            68,
            79,
            80,
            86,
        ],
        93: [
            4,
            8,
            25,
            47,
            53,
            56,
            64,
            71,

        ],
        94: [
            17,
            21,
            51,
            52,
            60,
            73,
            85,

        ],
        95: [
            27,
            70,
            78,
            82,
            84,
            88,

        ],
        96: [
            1,
            16,
            22,
            26,
            39,
            40,
            48,
            62,
            63,
            76,

        ],
        97: [
            2,
            12,
            13,
            20,
            32,
            45,
            50,
            59,
            69,
            81,
            87,

        ],
        98: [
            11,
            14,
            31,
            83,

        ],

    }


def gen_fonts():
    return {
        'font_color': 'white', # main color
        'font_header': ImageFont.truetype("Akrobat-Black.ttf", size=75), # header with label
        'font_mean': ImageFont.truetype("CoFoKak-Black.otf", size=75), # mean
        'font_values': ImageFont.truetype("arial.ttf", size=22), # for values in maps
        'font_legend': ImageFont.truetype("Akrobat-Black.ttf", size=22), # color rectangles legend
        'font_top_legend': ImageFont.truetype("Akrobat-Black.ttf", size=28), # TOP REGIONS label
        'font_top_regions': ImageFont.truetype("Akrobat-Black.ttf", size=26), # list of top regions
    }


def gen_color_palette():
    return {
        3: {
            'good': '#7f7f7f',
            'norm': '#a6a6a6',
            'bad': '#d9d9d9',
            'star': '#bfbfbf',
        },
        2: {
            'good': '#8c272d',
            'norm': '#ae676b',
            'bad': '#dcbec0',
            'star': '#bfbfbf',
        },
        1: {
            'good': '#012665',
            'norm': '#536a95',
            'bad': '#abb5ca',
            'star': '#bfbfbf',
        },
        4: {
            'good': '#0dcdcd',
            'norm': '#7ee2e3',
            'bad': '#b5eff0',
            'star': '#bfbfbf',
        },
        5: {
            'good': '#f7931c',
            'norm': '#f7b45b',
            'bad': '#fbce97',
            'star': '#bfbfbf',
        }
    }


def gen_reg_in_rf_coords():
    return {
         33: {'name': 'Мск',
          'coord': (540, 280, 640, 350),
          'coord_text': (550, 290),
          'coord_ball': (550, 320),
          'coord_delta': (630, 320)},
         35: {'name': 'Мурм',
          'coord': (300, 100, 400, 170),
          'coord_text': (310, 110),
          'coord_ball': (310, 140),
          'coord_delta': (390, 140)},
         54: {'name': 'Карел',
          'coord': (300, 180, 400, 250),
          'coord_text': (310, 190),
          'coord_ball': (310, 220),
          'coord_delta': (390, 220)},
         9: {'name': 'Волог',
          'coord': (410, 180, 510, 250),
          'coord_text': (420, 190),
          'coord_ball': (420, 220),
          'coord_delta': (500, 220)},
         29: {'name': 'ЛенОб',
          'coord': (190, 180, 290, 250),
          'coord_text': (200, 190),
          'coord_ball': (200, 220),
          'coord_delta': (280, 220)},
         38: {'name': 'Новгор',
          'coord': (300, 260, 400, 330),
          'coord_text': (310, 270),
          'coord_ball': (310, 300),
          'coord_delta': (390, 300)},
         67: {'name': 'СПб',
          'coord': (190, 260, 290, 330),
          'coord_text': (200, 270),
          'coord_ball': (200, 300),
          'coord_delta': (280, 300)},
         46: {'name': 'Псков',
          'coord': (190, 340, 290, 410),
          'coord_text': (200, 350),
          'coord_ball': (200, 380),
          'coord_delta': (280, 380)},
         18: {'name': 'Калин',
          'coord': (80, 340, 180, 410),
          'coord_text': (90, 350),
          'coord_ball': (90, 380),
          'coord_delta': (170, 380)},
         89: {'name': 'Яросл',
          'coord': (540, 200, 640, 270),
          'coord_text': (550, 210),
          'coord_ball': (550, 240),
          'coord_delta': (630, 240)},
         34: {'name': 'МосОб',
          'coord': (540, 360, 640, 430),
          'coord_text': (550, 370),
          'coord_ball': (550, 400),
          'coord_delta': (630, 400)},
         77: {'name': 'Тул',
          'coord': (540, 440, 640, 510),
          'coord_text': (550, 450),
          'coord_ball': (550, 480),
          'coord_delta': (630, 480)},
         30: {'name': 'Липецк',
          'coord': (540, 520, 640, 590),
          'coord_text': (550, 530),
          'coord_ball': (550, 560),
          'coord_delta': (630, 560)},
         5: {'name': 'Белгор',
          'coord': (540, 600, 640, 670),
          'coord_text': (550, 610),
          'coord_ball': (550, 640),
          'coord_delta': (630, 640)},
         75: {'name': 'Твер',
          'coord': (430, 280, 530, 350),
          'coord_text': (440, 290),
          'coord_ball': (440, 320),
          'coord_delta': (520, 320)},
         19: {'name': 'Калуж',
          'coord': (430, 360, 530, 430),
          'coord_text': (440, 370),
          'coord_ball': (440, 400),
          'coord_delta': (520, 400)},
         42: {'name': 'Орлов',
          'coord': (430, 440, 530, 510),
          'coord_text': (440, 450),
          'coord_ball': (440, 480),
          'coord_delta': (520, 480)},
         28: {'name': 'Курск',
          'coord': (430, 520, 530, 590),
          'coord_text': (440, 530),
          'coord_ball': (440, 560),
          'coord_delta': (520, 560)},
         72: {'name': 'Смол',
          'coord': (320, 360, 420, 430),
          'coord_text': (330, 370),
          'coord_ball': (330, 400),
          'coord_delta': (410, 400)},
         6: {'name': 'Брянск',
          'coord': (320, 440, 420, 510),
          'coord_text': (330, 450),
          'coord_ball': (330, 480),
          'coord_delta': (410, 480)},
         15: {'name': 'Иван',
          'coord': (650, 280, 750, 350),
          'coord_text': (660, 290),
          'coord_ball': (660, 320),
          'coord_delta': (740, 320)},
         24: {'name': 'Кост',
          'coord': (760, 280, 860, 350),
          'coord_text': (770, 290),
          'coord_ball': (770, 320),
          'coord_delta': (850, 320)},
         7: {'name': 'Влад',
          'coord': (650, 360, 750, 430),
          'coord_text': (660, 370),
          'coord_ball': (660, 400),
          'coord_delta': (740, 400)},
         65: {'name': 'Рязан',
          'coord': (650, 440, 750, 510),
          'coord_text': (660, 450),
          'coord_ball': (660, 480),
          'coord_delta': (740, 480)},
         74: {'name': 'Тамбов',
          'coord': (650, 520, 750, 590),
          'coord_text': (660, 530),
          'coord_ball': (660, 560),
          'coord_delta': (740, 560)},
         10: {'name': 'Ворон',
          'coord': (650, 600, 750, 670),
          'coord_text': (660, 610),
          'coord_ball': (660, 640),
          'coord_delta': (740, 640)},
         37: {'name': 'Нижег',
          'coord': (780, 380, 880, 450),
          'coord_text': (790, 390),
          'coord_ball': (790, 420),
          'coord_delta': (870, 420)},
         58: {'name': 'Мрд',
          'coord': (780, 460, 880, 530),
          'coord_text': (790, 470),
          'coord_ball': (790, 500),
          'coord_delta': (870, 500)},
         43: {'name': 'Пенз',
          'coord': (780, 540, 880, 610),
          'coord_text': (790, 550),
          'coord_ball': (790, 580),
          'coord_delta': (870, 580)},
         86: {'name': 'Чуваш',
          'coord': (890, 380, 990, 450),
          'coord_text': (900, 390),
          'coord_ball': (900, 420),
          'coord_delta': (980, 420)},
         57: {'name': 'Марий',
          'coord': (890, 300, 990, 370),
          'coord_text': (900, 310),
          'coord_ball': (900, 340),
          'coord_delta': (980, 340)},
         80: {'name': 'Ульян',
          'coord': (890, 460, 990, 530),
          'coord_text': (900, 470),
          'coord_ball': (900, 500),
          'coord_delta': (980, 500)},
         68: {'name': 'Сарат',
          'coord': (890, 540, 990, 610),
          'coord_text': (900, 550),
          'coord_ball': (900, 580),
          'coord_delta': (980, 580)},
         23: {'name': 'Киров',
          'coord': (1000, 300, 1100, 370),
          'coord_text': (1010, 310),
          'coord_ball': (1010, 340),
          'coord_delta': (1090, 340)},
         61: {'name': 'Татар',
          'coord': (1000, 380, 1100, 450),
          'coord_text': (1010, 390),
          'coord_ball': (1010, 420),
          'coord_delta': (1090, 420)},
         66: {'name': 'Самар',
          'coord': (1000, 460, 1100, 530),
          'coord_text': (1010, 470),
          'coord_ball': (1010, 500),
          'coord_delta': (1090, 500)},
         41: {'name': 'Орен',
          'coord': (1000, 540, 1100, 610),
          'coord_text': (1010, 550),
          'coord_ball': (1010, 580),
          'coord_delta': (1090, 580)},
         44: {'name': 'Перм',
          'coord': (1110, 300, 1210, 370),
          'coord_text': (1120, 310),
          'coord_ball': (1120, 340),
          'coord_delta': (1200, 340)},
         79: {'name': 'Удмур',
          'coord': (1110, 380, 1210, 450),
          'coord_text': (1120, 390),
          'coord_ball': (1120, 420),
          'coord_delta': (1200, 420)},
         49: {'name': 'Башк',
          'coord': (1110, 460, 1210, 530),
          'coord_text': (1120, 470),
          'coord_ball': (1120, 500),
          'coord_delta': (1200, 500)},
         8: {'name': 'Волгог',
          'coord': (780, 640, 880, 710),
          'coord_text': (790, 650),
          'coord_ball': (790, 680),
          'coord_delta': (870, 680)},
         53: {'name': 'Калм',
          'coord': (780, 720, 880, 790),
          'coord_text': (790, 730),
          'coord_ball': (790, 760),
          'coord_delta': (870, 760)},
         4: {'name': 'Астрах',
          'coord': (890, 720, 990, 790),
          'coord_text': (900, 730),
          'coord_ball': (900, 760),
          'coord_delta': (980, 760)},
         64: {'name': 'Ростов',
          'coord': (670, 720, 770, 790),
          'coord_text': (680, 730),
          'coord_ball': (680, 760),
          'coord_delta': (760, 760)},
         25: {'name': 'Красн',
          'coord': (560, 720, 660, 790),
          'coord_text': (570, 730),
          'coord_ball': (570, 760),
          'coord_delta': (650, 760)},
         47: {'name': 'Адыг',
          'coord': (450, 720, 550, 790),
          'coord_text': (460, 730),
          'coord_ball': (460, 760),
          'coord_delta': (540, 760)},
         56: {'name': 'Крым',
          'coord': (340, 720, 440, 790),
          'coord_text': (350, 730),
          'coord_ball': (350, 760),
          'coord_delta': (430, 760)},
         71: {'name': 'Севаст',
          'coord': (230, 720, 330, 790),
          'coord_text': (240, 730),
          'coord_ball': (240, 760),
          'coord_delta': (320, 760)},
         51: {'name': 'Дагест',
          'coord': (890, 820, 990, 890),
          'coord_text': (900, 830),
          'coord_ball': (900, 860),
          'coord_delta': (980, 860)},
         85: {'name': 'Чечня',
          'coord': (780, 820, 880, 890),
          'coord_text': (790, 830),
          'coord_ball': (790, 860),
          'coord_delta': (870, 860)},
         73: {'name': 'Ставр',
          'coord': (670, 820, 770, 890),
          'coord_text': (680, 830),
          'coord_ball': (680, 860),
          'coord_delta': (760, 860)},
         21: {'name': 'КЧР',
          'coord': (560, 820, 660, 890),
          'coord_text': (570, 830),
          'coord_ball': (570, 860),
          'coord_delta': (650, 860)},
         17: {'name': 'КБР',
          'coord': (560, 900, 660, 970),
          'coord_text': (570, 910),
          'coord_ball': (570, 940),
          'coord_delta': (650, 940)},
         60: {'name': 'Осетия',
          'coord': (670, 900, 770, 970),
          'coord_text': (680, 910),
          'coord_ball': (680, 940),
          'coord_delta': (760, 940)},
         52: {'name': 'Ингуш',
          'coord': (780, 900, 880, 970),
          'coord_text': (790, 910),
          'coord_ball': (790, 940),
          'coord_delta': (870, 940)},
         55: {'name': 'Коми',
          'coord': (1110, 200, 1210, 270),
          'coord_text': (1120, 210),
          'coord_ball': (1120, 240),
          'coord_delta': (1200, 240)},
         3: {'name': 'Арханг',
          'coord': (1000, 200, 1100, 270),
          'coord_text': (1010, 210),
          'coord_ball': (1010, 240),
          'coord_delta': (1090, 240)},
         36: {'name': 'НАО',
          'coord': (1110, 120, 1210, 190),
          'coord_text': (1120, 130),
          'coord_ball': (1120, 160),
          'coord_delta': (1200, 160)},
         84: {'name': 'Челяб',
          'coord': (1240, 460, 1340, 530),
          'coord_text': (1250, 470),
          'coord_ball': (1250, 500),
          'coord_delta': (1330, 500)},
         70: {'name': 'Сверд',
          'coord': (1240, 380, 1340, 450),
          'coord_text': (1250, 390),
          'coord_ball': (1250, 420),
          'coord_delta': (1330, 420)},
         82: {'name': 'ХМАО',
          'coord': (1240, 300, 1340, 370),
          'coord_text': (1250, 310),
          'coord_ball': (1250, 340),
          'coord_delta': (1330, 340)},
         88: {'name': 'ЯНАО',
          'coord': (1240, 220, 1340, 290),
          'coord_text': (1250, 230),
          'coord_ball': (1250, 260),
          'coord_delta': (1330, 260)},
         78: {'name': 'Тюмен',
          'coord': (1350, 300, 1450, 370),
          'coord_text': (1360, 310),
          'coord_ball': (1360, 340),
          'coord_delta': (1440, 340)},
         27: {'name': 'Курган',
          'coord': (1350, 380, 1450, 450),
          'coord_text': (1360, 390),
          'coord_ball': (1360, 420),
          'coord_delta': (1440, 420)},
         40: {'name': 'Омск',
          'coord': (1370, 480, 1470, 550),
          'coord_text': (1380, 490),
          'coord_ball': (1380, 520),
          'coord_delta': (1460, 520)},
         1: {'name': 'АлтКр',
          'coord': (1480, 480, 1580, 550),
          'coord_text': (1490, 490),
          'coord_ball': (1490, 520),
          'coord_delta': (1570, 520)},
         48: {'name': 'РесАлт',
          'coord': (1480, 560, 1580, 630),
          'coord_text': (1490, 570),
          'coord_ball': (1490, 600),
          'coord_delta': (1570, 600)},
         39: {'name': 'Нск',
          'coord': (1480, 400, 1580, 470),
          'coord_text': (1490, 410),
          'coord_ball': (1490, 440),
          'coord_delta': (1570, 440)},
         76: {'name': 'Томск',
          'coord': (1480, 320, 1580, 390),
          'coord_text': (1490, 330),
          'coord_ball': (1490, 360),
          'coord_delta': (1570, 360)},
         22: {'name': 'Кемер',
          'coord': (1590, 320, 1690, 390),
          'coord_text': (1600, 330),
          'coord_ball': (1600, 360),
          'coord_delta': (1680, 360)},
         26: {'name': 'КрасЯр',
          'coord': (1590, 240, 1690, 310),
          'coord_text': (1600, 250),
          'coord_ball': (1600, 280),
          'coord_delta': (1680, 280)},
         63: {'name': 'Хакас',
          'coord': (1590, 400, 1690, 470),
          'coord_text': (1600, 410),
          'coord_ball': (1600, 440),
          'coord_delta': (1680, 440)},
         62: {'name': 'Тыва',
          'coord': (1590, 480, 1690, 550),
          'coord_text': (1600, 490),
          'coord_ball': (1600, 520),
          'coord_delta': (1680, 520)},
         16: {'name': 'Иркут',
          'coord': (1700, 320, 1800, 390),
          'coord_text': (1710, 330),
          'coord_ball': (1710, 360),
          'coord_delta': (1790, 360)},
         50: {'name': 'Бурят',
          'coord': (1720, 420, 1820, 490),
          'coord_text': (1730, 430),
          'coord_ball': (1730, 460),
          'coord_delta': (1810, 460)},
         13: {'name': 'Забайк',
          'coord': (1720, 500, 1820, 570),
          'coord_text': (1730, 510),
          'coord_ball': (1730, 540),
          'coord_delta': (1810, 540)},
         12: {'name': 'ЕАО',
          'coord': (1830, 420, 1930, 490),
          'coord_text': (1840, 430),
          'coord_ball': (1840, 460),
          'coord_delta': (1920, 460)},
         45: {'name': 'Прим',
          'coord': (1940, 500, 2040, 570),
          'coord_text': (1950, 510),
          'coord_ball': (1950, 540),
          'coord_delta': (2030, 540)},
         2: {'name': 'Амур',
          'coord': (1830, 340, 1930, 410),
          'coord_text': (1840, 350),
          'coord_ball': (1840, 380),
          'coord_delta': (1920, 380)},
         59: {'name': 'Якут',
          'coord': (1830, 260, 1930, 330),
          'coord_text': (1840, 270),
          'coord_ball': (1840, 300),
          'coord_delta': (1920, 300)},
         32: {'name': 'Магад',
          'coord': (1940, 260, 2040, 330),
          'coord_text': (1950, 270),
          'coord_ball': (1950, 300),
          'coord_delta': (2030, 300)},
         87: {'name': 'Чукот',
          'coord': (1940, 180, 2040, 250),
          'coord_text': (1950, 190),
          'coord_ball': (1950, 220),
          'coord_delta': (2030, 220)},
         81: {'name': 'Хабар',
          'coord': (1940, 340, 2040, 410),
          'coord_text': (1950, 350),
          'coord_ball': (1950, 380),
          'coord_delta': (2030, 380)},
         69: {'name': 'Сахал',
          'coord': (2050, 340, 2150, 410),
          'coord_text': (2060, 350),
          'coord_ball': (2060, 380),
          'coord_delta': (2140, 380)},
         20: {'name': 'Камчат',
          'coord': (2050, 180, 2150, 250),
          'coord_text': (2060, 190),
          'coord_ball': (2060, 220),
          'coord_delta': (2140, 220)}
    }


def gen_fo_in_rf_coords():
    return {
        90: {
            'coord': (10, 215, 110, 315),
            'coord_text': (61, 225),
            'coord_delta': (65, 280)
        },
        91: {
            'coord': (10, 110, 110, 210),
            'coord_text': (61, 120),
            'coord_delta': (65, 175)
        },
        92: {
            'name': 'ПФО',
            'coord': (115, 215, 215, 315),
            'coord_text': (166, 225),
            'coord_delta': (170, 280)
        },
        93: {
            'name': 'ЮФО',
            'coord': (80, 320, 180, 420),
            'coord_text': (131, 330),
            'coord_delta': (135, 385)
        },
        94: {
            'name': 'СКФО',
            'coord': (80, 425, 180, 525),
            'coord_text': (131, 435),
            'coord_delta': (135, 490)
        },
        95: {
            'name': 'УрФО',
            'coord': (220, 215, 320, 315),
            'coord_text': (271, 225),
            'coord_delta': (275, 280)
        },
        96: {
            'name': 'СФО',
            'coord': (325, 215, 425, 315),
            'coord_text': (376, 225),
            'coord_delta': (380, 280)
        },
        97: {
            'name': 'ДФО',
            'coord': (430, 215, 530, 315),
            'coord_text': (481, 225),
            'coord_delta': (485, 280)
        },
        98: {
            'name': 'НР',
            'coord': (4, 200, 70, 315),
            'coord_text': (61, 225),
            'coord_delta': (65, 280)
        },
    }


def gen_rf_legend():
    return {
        # header, percentage rectangle and percentage mean value
        'mean':{
            'coord_rectangle': (1970, 8, 2200, 110),
            'coord_text': (1950, 15),
            'coord_value': (2005, 7)
        },
        # Top regions block
        'top_regions': {
            'coord_rectangle': (1590, 820, 2160, 1090),
            'coord_header': (1605, 830),
            0: (1605, 880),
            1: (1605, 920),
            2: (1605, 960),
            3: (1605, 1000),
            4: (1605, 1040)
        },
        # Color legend
        'legend':{
            'good':{
                'coord_rectangle': (1590, 680, 1660, 720),
                'coord_text': (1670, 688),
            },
            'avg': {
                'coord_rectangle': (1590, 720, 1660, 760),
                'coord_text': (1670, 728),
            },
            'bad': {
                'coord_rectangle': (1590, 760, 1660, 800),
                'coord_text': (1670, 768),
            }
        }
    }


def gen_fo_legend():
    return {
        # header, percentage rectangle and percentage mean value
        'mean': {
            'coord_rectangle': (550, 8, 673, 50),
            'coord_text': (515, 5),
            'coord_value': (595, 5)
        },
        # Top regions block
        'top_regions': {
            'coord_rectangle': (1590, 820, 2160, 1090),
            'coord_header': (1605, 830),
            0: (1605, 880),
            1: (1605, 920),
            2: (1605, 960),
            3: (1605, 1000),
            4: (1605, 1040)
        },
        # Color legend
        'legend': {
            'good': {
                'coord_rectangle': (220, 270, 270, 320),
                'coord_text': (280, 285),
            },
            'avg': {
                'coord_rectangle': (220, 320, 270, 370),
                'coord_text': (280, 335),
            },
            'bad': {
                'coord_rectangle': (220, 370, 270, 420),
                'coord_text': (280, 385),
            }
        }
    }