# validation sequences with interesting motions

selected_seqs = [
    "uniandes_basketball_004_45___1644___1818_start_0",
    "uniandes_basketball_003_41___21___369_start_0",
    "uniandes_basketball_001_24___1287___1446_start_0",
    "cmu_soccer16_2___3393___3540_start_0",
    "uniandes_basketball_004_24___1404___1575_start_0",
    "sfu_basketball012_11___417___615_start_0",
    "cmu_soccer16_2___5535___5739_start_0",
    "unc_basketball_03-30-23_01_30___90___1857_start_0",
    "iiith_soccer_031_6___15___4329_start_140",
    "iiith_soccer_015_2___51___750_start_80",
    "unc_basketball_03-30-23_01_24___1584___1803_start_0",
    "iiith_soccer_031_2___3___159_start_0",
    "sfu_basketball_09_32___1404___1608_start_20",
    "iiith_soccer_031_6___15___4329_start_940",
    "sfu_basketball_05_5___0___2046_start_480",
    "sfu_basketball_09_22___1575___1746_start_20",
    "unc_basketball_02-24-23_01_38___342___1245_start_40",
    "unc_basketball_02-24-23_01_2___714___2046_start_380",
    "utokyo_soccer_8000_43_2___858___4095_start_0",
    "upenn_0713_Dance_4_5___1887___4095_start_580",
    "utokyo_soccer_8000_43_2___858___4095_start_200",
    "cmu_soccer06_2___2673___3750_start_300",
    "iiith_cooking_60_2___3060___4095_start_220",
    "utokyo_soccer_8000_43_2___0___840_start_80",
    "cmu_soccer06_2___2673___3750_start_100",
    "utokyo_soccer_8000_43_2___858___4095_start_600",
    "georgiatech_cooking_13_02_4___49479___51123_start_20",
    "upenn_0713_Dance_4_2___0___4095_start_960",
    "iiith_cooking_58_2___2478___3498_start_160",
    "upenn_0713_Dance_5_6___2301___6798_start_0",
    "unc_soccer_09-21-23_01_18___4878___5040_start_20",
    "iiith_cooking_60_2___2391___2901_start_20",
    "upenn_0727_Partner_Dance_3_1_2___2502___3624_start_300",
    "iiith_cooking_58_4___0___156_start_0",
    "uniandes_dance_017_4___0___1989_start_0",
    "uniandes_cooking_004_6___13365___15651_start_220",
    "uniandes_cooking_006_6___3552___4062_start_40",
    "uniandes_cooking_004_6___13365___15651_start_620",
    "uniandes_cooking_004_6___8049___8817_start_120",
    "indiana_bike_10_8___198___486_start_0",
    "sfu_cooking_008_1___11454___14355_start_520",
    "indiana_bike_10_6___528___1518_start_0",
    "indiana_cooking_08_5___17859___18330_start_40",
    "georgiatech_cooking_06_02_2___27768___28932_start_60",
    "sfu_cooking_008_5___540___1233_start_160",
    "sfu_cooking_010_1___1368___2124_start_180",
    "georgiatech_cooking_13_02_4___15600___19455_start_800",
    "georgiatech_cooking_13_02_4___44043___45762_start_500",
    "unc_soccer_09-21-23_01_18___0___1905_start_340",
    "upenn_0711_Cooking_3_5___1092___1353_start_20",
    "indiana_cooking_16_2___11601___12882_start_200",
    "upenn_0702_Cooking_1_2___49167___49545_start_20",
    "uniandes_dance_017_6___0___1935_start_0",
    "upenn_0702_Cooking_1_3___23691___24849_start_320",
    "upenn_0711_Cooking_3_5___8919___9303_start_40",
    "indiana_cooking_08_5___25458___27054_start_240",
    "indiana_cooking_23_5___20049___22218_start_40",
    "unc_soccer_09-21-23_01_7___1041___1200_start_20",
    "nus_cpr_38_3___1701___1884_start_20",
    "uniandes_dance_017_6___0___1935_start_600",
    "minnesota_cooking_060_4___1749___2304_start_60",
    "georgiatech_bike_06_10___0___654_start_140",
    "minnesota_cooking_060_4___1026___1707_start_80",
    "unc_soccer_09-21-23_01_19___3237___3435_start_20",
    "minnesota_cooking_061_2___573___1104_start_120",
    "minnesota_cooking_060_2___0___1446_start_80",
    "utokyo_salad_11_1018_4___5820___9729_start_220",
    "sfu_cooking_010_1___10446___11013_start_0",
    "uniandes_dance_007_4___0___972_start_240",
    "utokyo_salad_11_1018_4___9750___10392_start_120",
    "indiana_bike_09_12___1128___1701_start_60",
    "georgiatech_bike_07_6___0___567_start_40",
    "utokyo_omelet_11_1001_4___5220___14478_start_2980",
    "utokyo_salad_11_1018_4___5820___9729_start_420",
    "indiana_bike_09_5___7611___7869_start_0",
    "georgiatech_bike_07_10___1779___2175_start_0",
    "georgiatech_bike_15_12___2481___3858_start_380",
    "nus_cpr_44_2___48___654_start_20",
    "indiana_music_04_4___0___16383_start_140",
    "nus_cpr_38_2___93___975_start_80",
    "indiana_music_04_4___0___16383_start_4540",
    "upenn_0722_Piano_1_7___504___702_start_20",
    "utokyo_cpr_2005_29_2___0___966_start_140",
    "nus_cpr_12_1___702___2046_start_0",
    "georgiatech_covid_06_4___729___8034_start_1840",
    "utokyo_cpr_2005_32_4___3024___4095_start_300",
    "georgiatech_covid_18_12___2595___3240_start_20",
    "utokyo_cpr_2005_30_2___627___1566_start_80",
    "utokyo_cpr_2005_32_4___144___993_start_40",
    "sfu_covid_004_2___2157___2940_start_20",
    "sfu_covid_006_6___0___4644_start_720",
    "indiana_music_03_4___1590___11568_start_3220",
    "georgiatech_covid_18_10___5610___8190_start_400",
    "georgiatech_covid_06_8___1764___7191_start_1140",
    "upenn_0722_Piano_1_6___21453___24954_start_1100",
    "indiana_music_04_4___0___16383_start_3540",
    "upenn_0724_Guitar_1_4___0___768_start_160",
    "sfu_covid_006_2___3639___4851_start_240",
    "upenn_0726_Duet_Violin_1_1_4___4650___5223_start_140",
    "sfu_covid_008_12___6690___7581_start_140",
]

selected_seq_names_start_idx = [
    ("uniandes_basketball_004_45___1644___1818", 0),
    ("uniandes_basketball_003_41___21___369", 0),
    ("uniandes_basketball_001_24___1287___1446", 0),
    ("cmu_soccer16_2___3393___3540", 0),
    ("uniandes_basketball_004_24___1404___1575", 0),
    ("sfu_basketball012_11___417___615", 0),
    ("cmu_soccer16_2___5535___5739", 0),
    ("unc_basketball_03-30-23_01_30___90___1857", 0),
    ("iiith_soccer_031_6___15___4329", 140),
    ("iiith_soccer_015_2___51___750", 80),
    ("unc_basketball_03-30-23_01_24___1584___1803", 0),
    ("iiith_soccer_031_2___3___159", 0),
    ("sfu_basketball_09_32___1404___1608", 20),
    ("iiith_soccer_031_6___15___4329", 940),
    ("sfu_basketball_05_5___0___2046", 480),
    ("sfu_basketball_09_22___1575___1746", 20),
    ("unc_basketball_02-24-23_01_38___342___1245", 40),
    ("unc_basketball_02-24-23_01_2___714___2046", 380),
    ("utokyo_soccer_8000_43_2___858___4095", 0),
    ("upenn_0713_Dance_4_5___1887___4095", 580),
    ("utokyo_soccer_8000_43_2___858___4095", 200),
    ("cmu_soccer06_2___2673___3750", 300),
    ("iiith_cooking_60_2___3060___4095", 220),
    ("utokyo_soccer_8000_43_2___0___840", 80),
    ("cmu_soccer06_2___2673___3750", 100),
    ("utokyo_soccer_8000_43_2___858___4095", 600),
    ("georgiatech_cooking_13_02_4___49479___51123", 20),
    ("upenn_0713_Dance_4_2___0___4095", 960),
    ("iiith_cooking_58_2___2478___3498", 160),
    ("upenn_0713_Dance_5_6___2301___6798", 0),
    ("unc_soccer_09-21-23_01_18___4878___5040", 20),
    ("iiith_cooking_60_2___2391___2901", 20),
    ("upenn_0727_Partner_Dance_3_1_2___2502___3624", 300),
    ("iiith_cooking_58_4___0___156", 0),
    ("uniandes_dance_017_4___0___1989", 0),
    ("uniandes_cooking_004_6___13365___15651", 220),
    ("uniandes_cooking_006_6___3552___4062", 40),
    ("uniandes_cooking_004_6___13365___15651", 620),
    ("uniandes_cooking_004_6___8049___8817", 120),
    ("indiana_bike_10_8___198___486", 0),
    ("sfu_cooking_008_1___11454___14355", 520),
    ("indiana_bike_10_6___528___1518", 0),
    ("indiana_cooking_08_5___17859___18330", 40),
    ("georgiatech_cooking_06_02_2___27768___28932", 60),
    ("sfu_cooking_008_5___540___1233", 160),
    ("sfu_cooking_010_1___1368___2124", 180),
    ("georgiatech_cooking_13_02_4___15600___19455", 800),
    ("georgiatech_cooking_13_02_4___44043___45762", 500),
    ("unc_soccer_09-21-23_01_18___0___1905", 340),
    ("upenn_0711_Cooking_3_5___1092___1353", 20),
    ("indiana_cooking_16_2___11601___12882", 200),
    ("upenn_0702_Cooking_1_2___49167___49545", 20),
    ("uniandes_dance_017_6___0___1935", 0),
    ("upenn_0702_Cooking_1_3___23691___24849", 320),
    ("upenn_0711_Cooking_3_5___8919___9303", 40),
    ("indiana_cooking_08_5___25458___27054", 240),
    ("indiana_cooking_23_5___20049___22218", 40),
    ("unc_soccer_09-21-23_01_7___1041___1200", 20),
    ("nus_cpr_38_3___1701___1884", 20),
    ("uniandes_dance_017_6___0___1935", 600),
    ("minnesota_cooking_060_4___1749___2304", 60),
    ("georgiatech_bike_06_10___0___654", 140),
    ("minnesota_cooking_060_4___1026___1707", 80),
    ("unc_soccer_09-21-23_01_19___3237___3435", 20),
    ("minnesota_cooking_061_2___573___1104", 120),
    ("minnesota_cooking_060_2___0___1446", 80),
    ("utokyo_salad_11_1018_4___5820___9729", 220),
    ("sfu_cooking_010_1___10446___11013", 0),
    ("uniandes_dance_007_4___0___972", 240),
    ("utokyo_salad_11_1018_4___9750___10392", 120),
    ("indiana_bike_09_12___1128___1701", 60),
    ("georgiatech_bike_07_6___0___567", 40),
    ("utokyo_omelet_11_1001_4___5220___14478", 2980),
    ("utokyo_salad_11_1018_4___5820___9729", 420),
    ("indiana_bike_09_5___7611___7869", 0),
    ("georgiatech_bike_07_10___1779___2175", 0),
    ("georgiatech_bike_15_12___2481___3858", 380),
    ("nus_cpr_44_2___48___654", 20),
    ("indiana_music_04_4___0___16383", 140),
    ("nus_cpr_38_2___93___975", 80),
    ("indiana_music_04_4___0___16383", 4540),
    ("upenn_0722_Piano_1_7___504___702", 20),
    ("utokyo_cpr_2005_29_2___0___966", 140),
    ("nus_cpr_12_1___702___2046", 0),
    ("georgiatech_covid_06_4___729___8034", 1840),
    ("utokyo_cpr_2005_32_4___3024___4095", 300),
    ("georgiatech_covid_18_12___2595___3240", 20),
    ("utokyo_cpr_2005_30_2___627___1566", 80),
    ("utokyo_cpr_2005_32_4___144___993", 40),
    ("sfu_covid_004_2___2157___2940", 20),
    ("sfu_covid_006_6___0___4644", 720),
    ("indiana_music_03_4___1590___11568", 3220),
    ("georgiatech_covid_18_10___5610___8190", 400),
    ("georgiatech_covid_06_8___1764___7191", 1140),
    ("upenn_0722_Piano_1_6___21453___24954", 1100),
    ("indiana_music_04_4___0___16383", 3540),
    ("upenn_0724_Guitar_1_4___0___768", 160),
    ("sfu_covid_006_2___3639___4851", 240),
    ("upenn_0726_Duet_Violin_1_1_4___4650___5223", 140),
    ("sfu_covid_008_12___6690___7581", 140),
]

# selected_seqs = [
#     "sfu_basketball_09_22___1575___1746_start_0",
#     "sfu_basketball_05_22___666___1371_start_0",
#     "cmu_soccer16_2___3393___3540_start_0",
#     "iiith_soccer_031_2___2367___3042_start_80",
#     "uniandes_basketball_001_27___1536___1716_start_0",
#     "iiith_soccer_031_2___3051___3858_start_60",
#     "uniandes_basketball_004_24___1212___1374_start_0",
#     "cmu_soccer16_2___5535___5739_start_0",
#     "unc_basketball_03-30-23_01_30___90___1857_start_0",
#     "iiith_soccer_031_2___1197___1875_start_40",
#     "iiith_soccer_031_2___3876___4095_start_0",
#     "sfu_basketball_05_22___666___1371_start_200",
#     "uniandes_basketball_004_45___1644___1818_start_20",
#     "unc_basketball_02-24-23_01_38___342___1245_start_40",
#     "sfu_basketball_09_22___711___897_start_20",
#     "unc_basketball_03-30-23_01_32___1461___1632_start_0",
#     "unc_basketball_02-24-23_01_39___1392___1530_start_0",
#     "utokyo_soccer_8000_43_2___858___4095_start_880",
#     "utokyo_soccer_8000_43_2___858___4095_start_80",
#     "utokyo_soccer_8000_43_2___0___840_start_160",
#     "utokyo_soccer_8000_43_2___858___4095_start_480",
#     "iiith_cooking_60_2___3060___4095_start_240",
#     "unc_soccer_09-21-23_01_18___0___1905_start_20",
#     "upenn_0713_Dance_4_2___0___4095_start_940",
#     "cmu_soccer16_2___6294___6525_start_40",
#     "uniandes_cooking_008_2___0___1362_start_200",
#     "iiith_cooking_58_2___0___2466_start_660",
#     "cmu_soccer06_3___0___3687_start_880",
#     "uniandes_basketball_003_42___585___825_start_20",
#     "sfu_cooking_008_5___2772___2973_start_20",
#     "unc_soccer_09-21-23_01_20___669___753_start_0",
#     "iiith_cooking_58_2___2478___3498_start_240",
#     "sfu_cooking_008_1___11454___14355_start_540",
#     "upenn_0713_Dance_5_6___2301___6798_start_340",
#     "upenn_0727_Partner_Dance_2_2_3___0___1479_start_440",
#     "minnesota_cooking_061_2___3333___9738_start_260",
#     "uniandes_cooking_007_8___6429___7497_start_100",
#     "unc_soccer_09-21-23_01_4___774___1023_start_20",
#     "upenn_0727_Partner_Dance_4_1_2___2850___3162_start_0",
#     "iiith_cooking_138_2___5478___6771_start_240",
#     "sfu_cooking_008_5___540___1233_start_40",
#     "uniandes_dance_019_47___0___2016_start_40",
#     "upenn_0711_Cooking_6_3___5961___7902_start_180",
#     "upenn_0702_Cooking_1_2___49167___49545_start_20",
#     "sfu_cooking_008_1___5373___7230_start_180",
#     "upenn_0702_Cooking_1_3___23691___24849_start_320",
#     "uniandes_dance_002_11___0___1629_start_300",
#     "indiana_cooking_23_3___16872___17940_start_300",
#     "indiana_cooking_23_5___20049___22218_start_40",
#     "indiana_cooking_09_2___4530___5337_start_120",
#     "indiana_cooking_23_3___1200___4059_start_780",
#     "minnesota_cooking_060_4___4332___8694_start_680",
#     "minnesota_cooking_060_4___1749___2304_start_120",
#     "uniandes_cooking_007_8___21399___22824_start_40",
#     "unc_soccer_09-21-23_01_19___3237___3435_start_20",
#     "uniandes_dance_017_6___0___1935_start_620",
#     "minnesota_cooking_061_2___1326___1908_start_0",
#     "uniandes_cooking_001_5___3651___4815_start_200",
#     "upenn_0711_Cooking_3_5___5265___7674_start_400",
#     "utokyo_salad_11_1018_4___9750___10392_start_140",
#     "uniandes_dance_007_5___0___1668_start_320",
#     "utokyo_salad_11_1018_4___5820___9729_start_440",
#     "indiana_bike_09_12___1128___1701_start_60",
#     "indiana_bike_10_2___0___1242_start_120",
#     "utokyo_salad_11_1018_4___546___1830_start_280",
#     "utokyo_salad_11_1018_4___5820___9729_start_1240",
#     "indiana_bike_10_6___528___1518_start_60",
#     "indiana_bike_09_8___0___522_start_0",
#     "georgiatech_bike_14_6___1146___2046_start_200",
#     "nus_cpr_44_2___897___1815_start_240",
#     "georgiatech_bike_15_6___531___2046_start_380",
#     "georgiatech_bike_15_2___0___1545_start_220",
#     "georgiatech_bike_16_14___5433___6969_start_120",
#     "utokyo_cpr_2005_32_4___144___993_start_120",
#     "georgiatech_cooking_13_02_2___7833___8670_start_160",
#     "indiana_music_04_4___0___16383_start_140",
#     "georgiatech_cooking_13_02_2___11268___12732_start_180",
#     "georgiatech_cooking_13_02_2___7509___7734_start_20",
#     "georgiatech_cooking_13_02_2___23721___25026_start_60",
#     "indiana_music_04_5___234___2160_start_0",
#     "indiana_music_04_4___0___16383_start_4540",
#     "utokyo_cpr_2005_29_2___0___966_start_140",
#     "georgiatech_covid_18_10___4098___5574_start_160",
#     "nus_cpr_44_3___1455___1767_start_0",
#     "utokyo_cpr_2005_32_4___1005___2148_start_240",
#     "nus_cpr_38_2___93___975_start_100",
#     "nus_cpr_13_1___483___2046_start_0",
#     "indiana_music_04_5___234___2160_start_400",
#     "sfu_covid_004_2___2157___2940_start_0",
#     "utokyo_cpr_2005_30_2___249___555_start_40",
#     "upenn_0722_Piano_1_6___21453___24954_start_1120",
#     "upenn_0724_Guitar_1_4___0___768_start_60",
#     "upenn_0726_Duet_Violin_1_1_3___243___4434_start_60",
#     "upenn_0724_Guitar_1_5___3045___3363_start_40",
#     "georgiatech_covid_06_8___1764___7191_start_1160",
#     "georgiatech_covid_04_8___5466___8190_start_560",
#     "sfu_covid_008_12___6690___7581_start_100",
#     "sfu_covid_008_16___5340___6024_start_180",
#     "sfu_covid_006_2___3639___4851_start_260",
#     "georgiatech_covid_02_4___3846___7371_start_1060",
# ]


def selected_seqs_fn():
    # Select sequences with good interesting motions for each scenario
    # and each location.
    from dataset.egoexo4d_take_dataset import EgoExo4D_Take_Dataset
    from utils.torch_utils import careful_collate_fn
    from tqdm.auto import tqdm
    import joblib
    from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset

    data_dir = "/vision/u/chpatel/data/egoexo4d_ee4d_motion"

    split = "val"
    ds = EE4D_Motion_Dataset(
        data_dir=data_dir,
        split=split,
        repre_type="v4_beta",
        cond_betas=False,
        cond_img_feat=False,
        cond_traj=True,
        img_feat_type="dinov2",
        do_normalization=True,
        window=80,
    )

    movement = {}

    for idx in tqdm(range(0, len(ds), 10)):
        batch = careful_collate_fn([ds[idx]])
        pred_mdata = ds.ret_to_full_sequence(batch)

        seq_name = batch["misc"]["seq_name"][0]
        start_idx = batch["misc"]["start_idx"][0] // 3
        k = f"{seq_name}_start_{start_idx}"

        kp3d = pred_mdata["kp3d"][0][:, :22]

        avg = kp3d.mean(0)
        diff = kp3d - avg[None]
        diff = diff.abs().mean()

        movement[k] = diff.item()
    # joblib.dump(movement, f"assets/movement_ee4d_val.pkl")
    # movement = joblib.load(f"assets/movement_ee4d_val.pkl")
    ls = list(zip(movement.keys(), movement.values()))
    ls.sort(key=lambda x: -x[1])

    eeds = EgoExo4D_Take_Dataset(data_dir=data_dir)

    done = {}
    selected = []
    seq_names_start_idx = []
    for seq_name, _ in ls:
        take_name = seq_name.split("___")[0]
        m = eeds.metadata[take_name]
        ptn = m["parent_task_name"]
        uni_name = m["university_name"]

        if (ptn, uni_name) not in done:
            done[(ptn, uni_name)] = 0

        if done[(ptn, uni_name)] >= 4:
            continue

        done[(ptn, uni_name)] += 1
        selected.append(seq_name)

        seq_name, start_idx = seq_name.split("_start_")
        start_idx = int(start_idx)
        seq_names_start_idx.append((seq_name, start_idx))

    print(selected)
    print(seq_names_start_idx)


# selected_seqs_fn()
