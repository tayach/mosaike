import pandapower as pp
import pandapower.networks as nw


def build_grid():
    """Create the distribution grid described in Section 3.2.2 of the thesis."""
    # The thesis specifies the Oberrhein MV network in Section 3.2.2.
    # pandapower provides a ready-to-use implementation which we utilise here.
    return nw.mv_oberrhein(scenario="load")


def main():
    net = build_grid()
    # Persist the created network for later use (see thesis Section 3.2.2).
    pp.to_json(net, "grid.json")


if __name__ == "__main__":
    main()
