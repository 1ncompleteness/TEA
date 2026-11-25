// Interactive tabs for the 33 v1.0.0 demo videos + copy-to-clipboard buttons
const RELEASE_BASE = "https://github.com/1ncompleteness/TEA/releases/download/v1.0.0/";

const VIDEOS = [
  // Habitat 3.0 Social-Nav (4)
  { suite: "habitat", tag: "Habitat 3.0 Social-Nav", title: "Episode 1 (two-part): DD-PPO K=1 vs. TEA K=4", file: "habitat3_social_nav__find_and_follow_the_humanoid.mp4" },
  { suite: "habitat", tag: "Habitat 3.0 Social-Nav", title: "Episode 2: TEA K=4 find & follow humanoid", file: "habitat3_social_nav__find_and_follow_the_humanoid_ep2.mp4" },
  { suite: "habitat", tag: "Habitat 3.0 Social-Nav", title: "Episode 3: TEA K=4 find & follow humanoid", file: "habitat3_social_nav__find_and_follow_the_humanoid_ep3.mp4" },
  { suite: "habitat", tag: "Habitat 3.0 Social-Nav", title: "Episode 4: TEA K=4 find & follow humanoid", file: "habitat3_social_nav__find_and_follow_the_humanoid_ep4.mp4" },

  // CALVIN Scene D (9)
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Press the button to turn on the LED", file: "calvin_scene_d__press_the_button_to_turn_on_the_led.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Lift the blue block from the table", file: "calvin_scene_d__lift_the_blue_block_from_the_table.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Lift the pink block from the table", file: "calvin_scene_d__lift_the_pink_block_from_the_table.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Lift the red block from the table", file: "calvin_scene_d__lift_the_red_block_from_the_table.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Push the blue block left", file: "calvin_scene_d__push_the_blue_block_left.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Push the pink block left", file: "calvin_scene_d__push_the_pink_block_left.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Push the red block left", file: "calvin_scene_d__push_the_red_block_left.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Push the red block right", file: "calvin_scene_d__push_the_red_block_right.mp4" },
  { suite: "calvin", tag: "CALVIN Scene D (Scripted Controller)", title: "Rotate the red block right", file: "calvin_scene_d__rotate_the_red_block_right.mp4" },

  // LIBERO-Long (10)
  { suite: "libero_10", tag: "LIBERO-Long", title: "Turn on the stove and put the moka pot on it", file: "libero_10__turn_on_the_stove_and_put_the_moka_pot_on_it.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put both moka pots on the stove", file: "libero_10__put_both_moka_pots_on_the_stove.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Pick up the book and place it in back compartment of caddy", file: "libero_10__pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put alphabet soup and cream cheese box in basket", file: "libero_10__put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put alphabet soup and tomato sauce in basket", file: "libero_10__put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put cream cheese box and butter in basket", file: "libero_10__put_both_the_cream_cheese_box_and_the_butter_in_the_basket.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put black bowl in bottom drawer of cabinet and close it", file: "libero_10__put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put white mug on left plate & yellow-white mug on right plate", file: "libero_10__put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put white mug on plate & chocolate pudding to right of plate", file: "libero_10__put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.mp4" },
  { suite: "libero_10", tag: "LIBERO-Long", title: "Put yellow and white mug in microwave and close it", file: "libero_10__put_the_yellow_and_white_mug_in_the_microwave_and_close_it.mp4" },

  // LIBERO-Goal (10)
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Open the middle drawer of the cabinet", file: "libero_goal__open_the_middle_drawer_of_the_cabinet.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Open the top drawer and put the bowl inside", file: "libero_goal__open_the_top_drawer_and_put_the_bowl_inside.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Push the plate to the front of the stove", file: "libero_goal__push_the_plate_to_the_front_of_the_stove.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the bowl on the plate", file: "libero_goal__put_the_bowl_on_the_plate.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the bowl on the stove", file: "libero_goal__put_the_bowl_on_the_stove.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the bowl on top of the cabinet", file: "libero_goal__put_the_bowl_on_top_of_the_cabinet.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the cream cheese in the bowl", file: "libero_goal__put_the_cream_cheese_in_the_bowl.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the wine bottle on the rack", file: "libero_goal__put_the_wine_bottle_on_the_rack.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Put the wine bottle on top of the cabinet", file: "libero_goal__put_the_wine_bottle_on_top_of_the_cabinet.mp4" },
  { suite: "libero_goal", tag: "LIBERO-Goal", title: "Turn on the stove", file: "libero_goal__turn_on_the_stove.mp4" },
];

function renderVideos(filter) {
  const grid = document.getElementById("video-grid");
  if (!grid) return;
  const items = filter === "all" ? VIDEOS : VIDEOS.filter((v) => v.suite === filter);
  grid.innerHTML = items.map((v) => {
    const url = RELEASE_BASE + v.file;
    return `
      <article class="video-card">
        <video controls preload="none" playsinline src="${url}"></video>
        <div class="video-meta">
          <div class="video-suite">${v.tag}</div>
          <div class="video-title">${v.title}</div>
          <a class="video-dl" href="${url}" target="_blank" rel="noopener">Download .mp4 &rarr;</a>
        </div>
      </article>
    `;
  }).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  renderVideos("habitat");

  document.querySelectorAll("[data-video-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-video-filter]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderVideos(btn.dataset.videoFilter);
    });
  });

  document.querySelectorAll("[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.getElementById(btn.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        const prev = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = prev; }, 1600);
      } catch (_) {}
    });
  });
});
