import pandas as pd
import numpy as np
import sarracen
import tempfile

from sarracen.readers.read_phantom_ev import (_infer_type,
                                              _determine_column_labels)


def test_determine_column_labels() -> None:
    """ Test that we can find the right column labels. """

    file = "# Not labels\n"
    file += "# [Also] [not] [labels]\n"
    file += "# [01 My column]\n"
    file += " 6\n"

    with tempfile.NamedTemporaryFile(mode="w+t") as fp:
        fp.write(file)
        fp.seek(0)

        labels = _determine_column_labels(fp)

        assert labels == ['My column']


def test_infer_types() -> None:
    """ Test type assignment for values in a Phantom .ev file. """

    values = {'5.2E-02': np.float64,
              '6': np.int32,
              'asdf': str,
              '0056': str}

    for k, v in values.items():
        assert isinstance(_infer_type(k), v)


def test_read_phantom_ev() -> None:
    """ Test reading a Phantom .ev file. """

    file = ("# [01        time]   [02        ekin]   [03      etherm]   "
            "[04        emag]   [05        epot]   [06        etot]   "
            "[07        erad]   [08      totmom]   [09      angtot]   "
            "[10     rho max]   [11     rho ave]   [12          dt]   "
            "[13   totentrop]   [14     rmsmach]   [15        vrms]   "
            "[16        xcom]   [17        ycom]   [18        zcom]   "
            "[19   alpha max]   [20    temp max]   [21    temp ave]   "
            "[22    temp min]   [23       B max]   [24       B ave]   "
            "[25       B min]   [26    divB max]   [27    divB ave]   "
            "[28 hdivB/B max]   [29 hdivB/B ave]   [30  beta_P max]   "
            "[31  beta_P ave]   [32  beta_P min]   [33      erot_x]   "
            "[34      erot_y]   [35      erot_z]   [36        erot]\n")
    file += ("  8.4970136193E+00   1.8177455909E-01   5.4158603628E-02   "
             "1.8595725176E-01  -5.0148412966E-01  -7.9593715176E-02   "
             "0.0000000000E+00   6.0650792798E-05   9.9442651708E-02   "
             "4.7623608015E-01   5.6566079437E-02   0.0000000000E+00   "
             "3.0692726829E-03   2.5968208333E+00   4.9629955791E-01   "
             "1.5594628940E-04   6.0505315025E-04  -4.0115705987E-04   "
             "1.0000000000E+00   4.1165935401E+02   1.3741297402E+02   "
             "1.3721978467E+01   2.2432062966E-01   4.3427317056E-02   "
             "5.1607751350E-03   1.8665742129E-02   1.1899320284E-03   "
             "1.2635612898E-01   2.8966833381E-03   1.3269416438E+01   "
             "2.2834704533E+00   9.3228152400E-02   4.5337249588E-03   "
             "4.5085730615E-03   4.5866997377E-03   7.8689076457E-03\n")
    file += ("  8.5636568633E+00   1.9333983767E-01   5.4158603628E-02   "
             "1.8765804711E-01  -5.1844445413E-01  -8.3287965715E-02   "
             "0.0000000000E+00   6.1844825030E-05   9.9421767848E-02   "
             "6.8487432502E-01   6.8034341066E-02   5.5536036727E-03   "
             "3.6803936979E-03   2.6790338479E+00   5.1184446728E-01   "
             "1.1994693751E-04   5.6833010193E-04  -4.0384702180E-04   "
             "1.0000000000E+00   4.1165935401E+02   1.3741297402E+02   "
             "1.3721978467E+01   2.8681787642E-01   4.7626622886E-02   "
             "5.0811583115E-03   2.1987430751E-02   1.4220720828E-03   "
             "1.3270508570E-01   2.9907607616E-03   1.3573672459E+01   "
             "2.2609329437E+00   8.6464621561E-02   5.2261830425E-03   "
             "5.1981031381E-03   4.9361014002E-03   8.8712097519E-03\n")
    file += ("  8.5858712780E+00   1.9758300444E-01   5.4158603628E-02   "
             "1.8827544891E-01  -5.2535612317E-01  -8.5339066199E-02   "
             "0.0000000000E+00   6.2229795234E-05   9.9416248567E-02   "
             "7.8044560235E-01   7.2825280935E-02   5.5536036727E-03   "
             "3.9356960963E-03   2.7085792747E+00   5.1743062654E-01   "
             "1.2013369868E-04   4.9536590288E-04  -3.6832589199E-04   "
             "1.0000000000E+00   4.1165935401E+02   1.3741297402E+02   "
             "1.3721978467E+01   3.1031789206E-01   4.9249882671E-02   "
             "5.0559812966E-03   2.3748042062E-02   1.5435300935E-03   "
             "1.3326222708E-01   3.0514621694E-03   1.3671177206E+01   "
             "2.2532151373E+00   8.4221852421E-02   5.5003216703E-03   "
             "5.4711619748E-03   5.0658397533E-03   9.2655212610E-03\n")

    columns = ['time', 'ekin', 'etherm', 'emag', 'epot', 'etot', 'erad',
               'totmom', 'angtot', 'rho max', 'rho ave', 'dt', 'totentrop',
               'rmsmach', 'vrms', 'xcom', 'ycom', 'zcom', 'alpha max',
               'temp max', 'temp ave', 'temp min', 'B max', 'B ave', 'B min',
               'divB max', 'divB ave', 'hdivB/B max', 'hdivB/B ave',
               'beta_P max', 'beta_P ave', 'beta_P min', 'erot_x', 'erot_y',
               'erot_z', 'erot']

    with tempfile.NamedTemporaryFile(mode="w+t") as fp:
        fp.write(file)
        fp.seek(0)

        df = sarracen.read_phantom_ev(fp.name)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df.columns.tolist() == columns
        assert (df.dtypes == np.float64).all()
        assert df['erad'].iloc[0] == 0
        assert df['alpha max'].iloc[2] == 1.0


def test_read_phantom_ev_apr() -> None:
    """ Test reading a Phantom APR .ev file. """

    file = "# APR info for region   1\n"
    file += ("# [01        time]   [02        dump]   [03    x centre]   "
             "[04    y centre]   [05    z centre]   [06 radius_1   ]   "
             "[07 radius_2   ]   [08 radius_3   ]\n")
    file += ("  1.9672162700E+00             00001   -9.9711852483E+00  "
             "-6.0707469959E-01  -6.7316720682E-04   5.0000000000E+00   "
             "4.0000000000E+00   3.0000000000E+00\n")
    file += ("  3.9344325400E+00             00002   -9.9106996478E+00  "
             "-1.2298956402E+00  -2.9168380415E-03   5.0000000000E+00   "
             "4.0000000000E+00   3.0000000000E+00")

    columns = ['time', 'dump', 'x centre', 'y centre', 'z centre',
               'radius_1', 'radius_2', 'radius_3']

    with tempfile.NamedTemporaryFile(mode="w+t") as fp:
        fp.write(file)
        fp.seek(0)

        df = sarracen.read_phantom_ev(fp.name)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.columns.tolist() == columns
        assert df['dump'].iloc[0] == '00001'
        assert df['dump'].iloc[1] == '00002'
        assert df['radius_1'].iloc[0] == 5.0
        assert df['radius_2'].iloc[1] == 4.0


def test_read_phantom_ev_sink() -> None:
    """ Test reading a Phantom sink .ev file. """

    file = ("# [01        time]   [02           x]   [03           y]   "
            "[04           z]   [05        mass]   [06          vx]   "
            "[07          vy]   [08          vz]   [09       spinx]   "
            "[10       spiny]   [11       spinz]   [12        macc]   "
            "[13          fx]   [14          fy]   [15          fz]   "
            "[16        fssx]   [17        fssy]   [18        fssz]   "
            "[19     sink ID]   [20     nptmass]\n")
    file += ("   1.094425338E-02   -1.997268229E+01   -2.344990793E-03   "
             "-8.765034088E-05    9.547657483E-04    2.646004278E-05   "
             "-2.237704046E-01    2.716904788E-08    4.356594234E-11    "
             "1.878688488E-10    2.802086197E-10    5.000000000E-07    "
             "2.525195354E-03   -2.236358497E-06    2.514080724E-06    "
             "2.509542638E-03    3.058325956E-07   -3.135334592E-15    "
             "              3                  3\n")
    file += ("2.188850676E-02   -1.997268185E+01   -4.793990927E-03   "
             "-8.764989296E-05    9.547657483E-04    5.409642132E-05   "
             "-2.237704274E-01    5.468461538E-08    4.356594234E-11    "
             "1.878688488E-10    2.802086197E-10    5.000000000E-07    "
             "2.525195488E-03   -1.929268523E-06    2.514233371E-06    "
             "2.509571612E-03    5.985744372E-07    1.104006904E-08    "
             "              3                  3")

    columns = ['time', 'x', 'y', 'z', 'mass', 'vx', 'vy', 'vz', 'spinx',
               'spiny', 'spinz', 'macc', 'fx', 'fy', 'fz', 'fssx', 'fssy',
               'fssz', 'sink ID', 'nptmass']

    with tempfile.NamedTemporaryFile(mode="w+t") as fp:
        fp.write(file)
        fp.seek(0)

        df = sarracen.read_phantom_ev(fp.name)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.columns.tolist() == columns
        assert df['macc'].iloc[0] == 5e-7
        assert df['sink ID'].iloc[0] == 3
        assert df['nptmass'].iloc[1] == 3
