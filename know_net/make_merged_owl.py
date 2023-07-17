from typing import List
import os

from know_net.owl_maker import ONT


def main() -> None:
    files = os.listdir("data")
    turtles: List[str] = [ONT]
    for file in files:
        with open(f"data/{file}", "r") as f:
            turtle = f.read()
        turtles.append(turtle)
    with open("ontology.owl", "w") as f:
        f.write("\n".join(turtles))


if __name__ == "__main__":
    main()
