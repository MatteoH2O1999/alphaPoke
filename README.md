# alphaPoke
[![Tests](https://github.com/MatteoH2O1999/alphaPoke/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/MatteoH2O1999/alphaPoke/actions/workflows/test_workflow.yml)
[![Build](https://github.com/MatteoH2O1999/alphaPoke/actions/workflows/build_workflow.yml/badge.svg)](https://github.com/MatteoH2O1999/alphaPoke/actions/workflows/build_workflow.yml)
[![codecov](https://codecov.io/gh/MatteoH2O1999/alphaPoke/branch/main/graph/badge.svg?token=21UL1WOUAC)](https://codecov.io/gh/MatteoH2O1999/alphaPoke)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A pokémon showdown battle-bot project based on reinforcement learning techniques.
It includes an easy-to-use GUI interface to challenge the created models.
Keep in mind that this is a project I'm doing for a uni course.
As such, usage of other files is not officially supported and may change unexpectedly.

## Installation
The software does not require an installation process.
Download the correct file from the [release](https://github.com/MatteoH2O1999/alphaPoke/releases) section.
How to start it depends on your OS:

### Windows
Just download and open `alphaPoke_win.exe`.

Your browser will probably tell you that the file cannot be trusted.
Most browsers just give you the option to download anyway.

When you start it for the first time `Windows SmartScreen` will probably pop out.
Just click on `More info` and then `Run anyway`.

### MacOS
Download `alphaPoke_macOS.zip` and unpack it.
Inside there is the app ready to go.
MacOS will probably tell you that the app is from an untrusted source.
To open it anyway just right-click it and click on `open`.

### Linux
Download `alphaPoke_linux` and put it where you want.
Then add the executable bit to the file with `chmod +x path/to/executable`. 
At this point you can start it with `path/to/executable`.

## Getting started

### What do I need?

- 2 Pokémon showdown accounts
- That's it

For the sake of the tutorial we'll assume you have the following 2 accounts:
- Username: `account1`, Password: `password1`
- Username: `account2`, Password: `password2`

### Step 1: Showdown login
In a browser open [pokémon showdown](https://play.pokemonshowdown.com/) and login with one of your accounts.
In this case we will log in with `account1`, `password1`.

### Step 2: Setup bot account
The first step to challenge a bot is to set up its showdown account.
With the `alphaPoke` application open just type the bot account information into the relative fields (they are the first ones from the top).

So in the frame `Bot account info` we will type `account2` into the `Username:` field and `password2` into the `Password:` field.

### Step 3: Choose opponent type and battle format
In the frame `Choose your opponent` you will find a dropdown menu with all the playable agents.
Choosing one will print a short description in the box underneath and unlock the battle format selection.
The battle format selection menu is updated in real time and contains all the formats the agent you selected is capable of playing.
Choose the battle format you want.

### Step 4: Setup challenge account
In the last frame, the one titled `Challenge controls` you want to choose whether the agent will activate the battle timer or not.
As a last step you need to type the username that the bot will challenge.

In our case, we will type `account1`.

### Step 5: Start challenge
Now everything is set up and the only thing to do is starting the challenge.
On the bottom of the window you'll see two buttons: one for sending the challenge and one for accepting it.

#### Option 1: Send challenge
If the account the bot is using is capable of sending challenges, this is the easiest option:
once you click the button you will receive a challenge from the bot.

In our case if we click the button `Send challenge to account1` we will receive a challenge from `account2` in our browser, where we are logged in with `account 1`

> :warning: **Warning:** if you try to send a challenge to an account without being logged in the challenge will be lost.

#### Option 2: Accept challenge
If the account the bot is using is not capable of sending challenges, or for whatever reason you prefer being the one challenging,
you can certainly do this with the `Accept challenge` button.

In our case , clicking the `Accept challenge from account1` the bot will wait for a challenge from `account1` and you just have to send it from the browser.

> :warning: **Warning:** the account you are using in the browser needs to be able to challenge a player (that is a limitation for new accounts. Just use them a bit and it will unlock).

### Step 6: Glhf
Enjoy the bot.

glhf.