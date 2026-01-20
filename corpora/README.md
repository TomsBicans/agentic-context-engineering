
# Proof of concept test dataset

This page: https://en.wikipedia.org/wiki/Solar_System

Saved on a browser with `Save as` feature, then put in the corpora directory so it looks something like this:
```
(.venv) PS C:\Users\tomsb\Documents\GIT_REPOS\master-thesis\agentic-context-engineering\corpora> tree /F
Folder PATH listing for volume OS
C:.
│   README.md
│
└───scraped_data
    └───solar_system_wiki
        │   Solar System - Wikipedia.html
        │   Solar System - Wikipedia.md
        │   Solar System - Wikipedia.txt
        │
        └───Solar System - Wikipedia_files
                0355a973ffa402dc57f8f4371f702db85b17e989
                120px-SaganWalk.0.Sun.jpg
                250px-The_Four_Largest_Asteroids.jpg
                45e5789e5d9c8f7c79744f43ecaaf8ba42a8553a
                5a386d5764fd35c853376fd570d4c46300b19867
                876ecaef49f096f44b57f0258336275f8ba3a373
                Apparent_retrograde_motion_of_Mars_in_2003.gif
                Comet_Hale-Bopp_1995O1.jpg
                Commons-logo.svg.png
                Crab_Nebula.jpg
                Cscr-featured.svg.png
                db73983682cbebba39553ac1760903b39e050466
                Diagram_of_different_habitable_zone_regions_by_Chester_Harman.jpg
                Distant_object_orbits_+_Planet_Nine.png
                ea2097c0262c82b8e921dfcc2cc9873e238bc31c
                Gnome-mime-sound-openclipart(1).svg.png
                Gnome-mime-sound-openclipart.svg.png
                He1523a.jpg
                Inner_solar_system_objects_top_view_for_wiki.png
                Interstellar_medium_annotated.jpg
                Kuiper_belt_-_Oort_cloud-en.svg.png
                Kuiper_belt_plot_objects_of_outer_solar_system.png
                load(1).php
                load(2).php
                load.php
                mediawiki_compact.svg
                Meteor_shower_in_the_Chilean_Desert_(annotated_and_cropped).jpg
                Milky_Way_side_view.png
                OOjs_UI_icon_edit-ltr-progressive.svg.png
                Orbital_distances_in_the_solar_system_linear_scale.png
                PD-icon.svg.png
                PIA21424_-_The_TRAPPIST-1_Habitable_Zone.jpg
                PIA22835-VoyagerProgram&Heliosphere-Chart-20181210.png
                Planet_collage_to_scale_(captioned).jpg
                Plane_of_Ecliptic.jpg
                RocketSunIcon.svg.png
                Semi-protection-shackle.svg.png
                Solar_system.jpg
                Solar_system_delta_v_map.svg.png
                Solar_System_distance_to_scale.svg.png
                Solar_System_Missions.png
                Solar_system_orrery_inner_planets.gif
                Solar_system_orrery_outer_planets.gif
                Solar_System_Template_2.png
                Solar_System_true_color_(title_and_caption).jpg
                Soot-line1.jpg
                Sun_red_giant.svg.png
                Symbol_category_class.svg.png
                Symbol_portal_class.svg.png
                Terrestrial_planet_sizes_3.jpg
                TheKuiperBelt_classes-en.svg.png
                TheKuiperBelt_Projections_100AU_Classical_SDO.svg.png
                The_Earth_seen_from_Apollo_17_with_transparent_background.png
                The_Local_Interstellar_Cloud_and_neighboring_G-cloud_complex.svg.png
                The_Solar_System,_with_the_orbits_of_5_remarkable_comets._LOC_2013593161_(cropped).jpg
                The_Sun_in_white_light.jpg
                Wikibooks-logo.svg.png
                Wikidata-logo.svg.png
                wikimedia.svg
                Wikinews-logo.svg.png
                wikipedia-tagline-en.svg
                wikipedia-wordmark-en.svg
                wikipedia.png
                Wikiquote-logo.svg.png
                Wikisource-logo.svg.png
                Wikiversity_logo_2017.svg.png
                Wiktionary-logo-v2.svg.png

(.venv) PS C:\Users\tomsb\Documents\GIT_REPOS\master-thesis\agentic-context-engineering\corpora> 
```

The .txt file is just the .html file copied with manually changed extension. The .md file is created using this tool, where input html is copied in and md content is generated out: https://codebeautify.org/html-to-markdown