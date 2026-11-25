# Frog ID (prototype)

Identify East Texas frogs and toads by their calls, in the browser:
**https://lukbio.github.io/frog-id-web/**

- **Identify**: tap the microphone and hold your phone up to a chorus. Every 3
  seconds it names every frog that passes the confidence setting (or says
  "not sure" / "no frog"), with a live spectrogram and a list of what it heard
  this session. Or analyze a recording (m4a, mp3, wav) or play a sample. The
  home screen shows a frog of the day.
- **Explore**: the 24 East Texas frogs and toads (23 in the model; the Houston
  Toad has too few recordings), with photos, search, and month bars showing
  when each one calls. Frogs likely near you this month come first. Each frog
  has its own page: swipe through photos, play its calls, see when it calls,
  read about it (from Wikipedia), see a range map, share it, or add it to your
  Life List.
- **Life List**: frogs you've found (open a frog, tap Add to Life List), plus
  the ones likely nearby that you haven't found yet. Saved on the phone only.
- **Settings**: confidence, the location and season filter (your location is
  rounded and kept on your phone), scientific names, and offline status.

Photos come from iNaturalist (Creative Commons-licensed photos only, credited
on each frog's page). They load the first time you open the app with signal.

Everything runs on the device. Audio is never uploaded. After the first visit
it works with no signal, and Safari's Share → Add to Home Screen turns it into
an app icon.

## How it works

Each 3-second window (48 kHz mono) goes through BirdNET V2.4, cut down to its
1,024-number embedding (`model/birdnet_embed.tflite`, run with TensorFlow Lite
for the web). A logistic-regression classifier (`model/frog_classifier.json`)
trained on about 5,000 iNaturalist frog recordings, plus insects, night birds,
rain, wind and traffic, gives each species its own score. `model/filter.json` holds
the range grid (half-degree cells) and calling months.

In cross-validated testing it named the right species 74% of the time on the
first guess, and was right 90% of the time when at least 70% confident (47% of
recordings). It also trains on made-up mixes of two frogs, so it can name more
than one; on test mixes it names both 16% of the time, so choruses are still
its weak spot.

## Credits and licences

- BirdNET V2.4 model by the K. Lisa Yang Center for Conservation Bioacoustics
  (Cornell Lab of Ornithology) and Chemnitz University of Technology,
  CC BY-NC-SA 4.0 (`model/BIRDNET_LICENSE.txt`).
- Training and sample recordings: iNaturalist contributors (credited in the
  app for each sample) and ESC-50 (K. J. Piczak, 2015), CC BY-NC.
- Range and season data: iNaturalist research-grade observations.
- TensorFlow.js and TFLite for the web (`vendor/`): Google, Apache 2.0.

Non-commercial research and education use only.
