"""Provide unit tests for distribution_likelihood."""

from idtxl.data import Data
import numpy as np
import pytest
from idtxl.get_distribution_likelihood import get_distribution_likelihood

# settings
single = "norm"
single_err = "abcd"
shortlist = ["chi2", "laplace", "norm"]
longlist = ["chi2", "laplace", "norm", "triang", "logistic", "levy"]
list_err = ["chi2", "abcd", "norm"]


def create_data(processes, replications, samples=200):
    # create data with Process 0: normal distribution and Process 1: chi2 distribution
    mu = 0
    sigma = 0.1
    dat = np.random.normal(mu, sigma, size=(processes, samples, replications))
    if processes > 1:
        chi = np.random.chisquare(df=2, size=(1, samples, replications))
        dat[1, :, :] = chi
    data = Data(normalise=False)  # initialise an empty data object
    data.set_data(dat, "psr")

    return data


def test_input_mode():
    """ Tests different and wrong input for mode """

    # create data
    data = create_data(2, 3)

    # test wrong inputs
    with pytest.raises(RuntimeError):
        dist_like = get_distribution_likelihood(data, distributions=single_err)
        dist_like.fit(mode="abc")

    # test mode per_replication
    mode = "per_replication"
    dist_like = get_distribution_likelihood(data, distributions=single)
    dl = dist_like.fit(mode=mode)
    assert (dl.n_data_replications == data.n_replications), \
        "Number of replications are not correctly set in distribution_likelihood object."
    assert (dl.n_data_processes == data.n_processes), \
        "Number of processes are not correctly set in distribution_likelihood object."
    assert (dl.mode == mode), \
        "mode is not correctly set in distribution_likelihood object."
    assert (len(dl.results) == data.n_processes), \
        f"Number of processes in summary {len(dl.results)} differ from the defined processes 2."
    assert (len(dl.results) == data.n_processes), \
        f"Number of processes in summary {len(dl.results)} differ from the defined processes 2."
    for p in range(data.n_processes):
        assert (len(dl.results[p]) == data.n_replications), \
            f"Number of replications differ between summary {len(dl.results[p])} and data {data.n_replications}."

    # test mode over_all_replications
    mode = "over_all_replications"

    dist_like = get_distribution_likelihood(data, distributions=single)
    dl = dist_like.fit(mode=mode)
    assert (dl.n_data_replications == data.n_replications), \
        "Number of replications are not correctly set in distribution_likelihood object."
    assert (dl.n_data_processes == data.n_processes), \
        "Number of processes are not correctly set in distribution_likelihood object."
    assert (dl.mode == mode), \
        "mode is not correctly set in distribution_likelihood object."
    for p in range(data.n_processes):
         assert (isinstance(dl.results[p], dict)), \
             f"Incorrect type of summary {type(dl.results[p])}."


def test_input_dist():
    """ Tests different and wrong input for dist """

    # create data
    data = create_data(1, 3)

    mode = "over_all_replications"

    # test wrong inputs
    with pytest.raises(RuntimeError):
        dist_like = get_distribution_likelihood(data, distributions=single_err)
        dist_like.fit(mode=mode)
    with pytest.raises(RuntimeError):
        dist_like = get_distribution_likelihood(data, distributions=list_err)
        dist_like.fit(mode=mode)

    # test dist input types
    dist_like = get_distribution_likelihood(data, distributions=single)
    dl = dist_like.fit(mode=mode)
    assert (len(dl.results[0]["summary"].index) == 1), \
        f"Numbers of tested distributions differ between input: 1 and distribution likelihood" \
        f"object {len(dl.results[0]['summary'].index)}"

    dist_like = get_distribution_likelihood(data, distributions=shortlist)
    dl = dist_like.fit(mode=mode)
    assert (len(dl.results[0]["summary"].index) == len(shortlist)), \
        f"Numbers of tested distributions differ between input: {len(shortlist)} and " \
        f"distribution_likelihood object {len(dl.results[0]['summary'].index)}."

    dist_like = get_distribution_likelihood(data, distributions="common")
    dl = dist_like.fit(mode=mode)
    dist_like = get_distribution_likelihood(data, distributions="all")
    dl = dist_like.fit(mode=mode)


def test_input_processes():
    """ Tests different and wrong input for processes """

    # create data
    data = create_data(3, 2, samples=500)

    mode = "over_all_replications"

    dist_like = get_distribution_likelihood(data, distributions=single)

    # test wrong inputs
    with pytest.raises(RuntimeError):
        dist_like.fit(mode=mode, processes=5)
    with pytest.raises(RuntimeError):
        dist_like.fit(mode=mode, processes=[1, 5])
    with pytest.raises(RuntimeError):
        dist_like.fit(mode=mode, processes="abc")

    # test single process
    dl = dist_like.fit(mode=mode, processes=1)
    assert (dl.results[0] == "Not tested" and dl.results[2] == "Not tested"), \
        "tested process differs from the process in summary."
    assert (isinstance(dl.results[1], dict)), \
        f"Incorrect type of summary {type(dl.results[1])}."

    # test process list
    dl = dist_like.fit(mode=mode, processes=[0, 2])

    assert (dl.results[1] == "Not tested"), \
        "tested process differs from the process in summary."
    assert (isinstance(dl.results[0], dict) and isinstance(dl.results[2], dict)), \
        f"Incorrect input in process 0 and/or 2."


def test_output():
    """ Tests if the correct dists are found for the test data """

    # create data
    data = create_data(2, 2, samples=800)

    mode = "over_all_replications"

    dist_like = get_distribution_likelihood(data, distributions=longlist)
    
    # test correct output of the test data
    dl = dist_like.fit(mode=mode)
    d0 = dl.get_all_dists(0)
    d1 = dl.get_all_dists(1)
    assert (dl.get_best_dist(0) == "norm" or "norm" in d0[0:2]), \
        f"For process 0 the wrong distribution was fount {dl.get_best_dist(0)}. Should be \"norm\"."
    assert (dl.get_best_dist(1) == "chi2" or "chi2" in d1[0:2]), \
        f"For process 0 the wrong distribution was fount {dl.get_best_dist(0)}. Should be \"chi2\"."


def test_differential_entropy():
    """ Tests calculation of differential entropy """

    # create data
    data = create_data(2, 3)

    dist_like = get_distribution_likelihood(data, distributions="common")
    mode = "over_all_replications"
    dl = dist_like.fit(mode=mode)
    de = dl.differential_entropy(0)
    a=1

if __name__ == '__main__':
    #test_input_mode()
    #test_input_dist()
    #test_input_processes()
    #test_output()

    test_differential_entropy()
