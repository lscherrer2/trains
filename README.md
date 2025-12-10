UV Package manager recommended to reproduce dev results.

Clone this repo & install into your local environment with `./script/install.sh`
Run tests with `./script/test.sh`

This package simulates a train track system. It is designed for one that's densely connected.

A `System` can be initialized from a `JSON` so as to not have to fuss with the python objects.
You can figure it out just looking at the simple test case in `test/data/system.json`

The setup centers around switches (`Switch`). Each switch has 3 branches (`Branch`): `approach`, 
`through`, and `diverging`. Branches are connected to each other with tracks (`Track`).

Trains are simulated in the systme by calling their `.step(dt)` method.
