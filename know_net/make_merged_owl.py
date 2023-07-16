from typing import List
from know_net.owl_maker import ONT
import os


def main() -> None:
    files = os.listdir("data")
    turtles: List[str] = [ONT]
    for file in files:
        with open(f"data/{file}", "r") as f:
            turtle = f.read()
        turtles.extend(turtle)
    with open("ontology.owl", "w") as f:
        f.write("\n".join(turtles))
if __name__ == "__main__":
    main()
