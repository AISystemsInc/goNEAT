package genetics

import (
	"github.com/AISystemsInc/goNEAT/v2/neat"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"log"
	"math/rand"
	"testing"
	"time"
)

func sequentialExecutorNextEpoch(pop *Population, opts *neat.Options) error {
	ex := SequentialPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(opts.NeatContext(), i+1, pop)
		if err != nil {
			return errors.Wrapf(err, "failed at: %d epoch", i)
		}
	}
	return nil

}

func parallelExecutorNextEpoch(pop *Population, opts *neat.Options) error {
	ex := ParallelPopulationEpochExecutor{}

	for i := 0; i < 100; i++ {
		err := ex.NextEpoch(opts.NeatContext(), i+1, pop)
		if err != nil {
			return errors.Wrapf(err, "failed at: %d epoch", i)
		}
	}
	return nil
}

func TestPopulationEpochExecutor_NextEpoch(t *testing.T) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 15, 3
	linkProb := 0.8
	conf := neat.Options{
		CompatThreshold: 0.5,
		DropOffAge:      1,
		PopSize:         30,
		BabiesStolen:    10,
		RecurOnlyProb:   0.2,
	}
	neat.LogLevel = neat.LogLevelInfo
	gen := newGenomeRand(1, in, out, n, nmax, false, linkProb)
	pop, err := NewPopulation(gen, &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	// test sequential executor
	err = sequentialExecutorNextEpoch(pop, &conf)
	assert.NoError(t, err, "failed to run sequential epoch executor")

	// test parallel executor
	err = parallelExecutorNextEpoch(pop, &conf)
	assert.NoError(t, err, "failed to run parallel epoch executor")
}

func BenchmarkPopulationEpochExecutor_NextEpoch(b *testing.B) {
	rand.Seed(42)
	in, out, nmax, n := 3, 2, 15, 3
	linkProb := 0.8

	b.Run("sequential", func(b *testing.B) {
		var start = time.Now()
		conf := neat.Options{
			CompatThreshold: 0.5,
			DropOffAge:      1,
			PopSize:         200,
			BabiesStolen:    10,
			RecurOnlyProb:   0.2,
		}
		neat.LogLevel = neat.LogLevelInfo
		gen := newGenomeRand(1, in, out, n, nmax, false, linkProb)
		pop, err := NewPopulation(gen, &conf)
		require.NoError(b, err, "failed to create population")
		require.NotNil(b, pop, "population expected")
		for i := 0; i < b.N; i++ {
			err = sequentialExecutorNextEpoch(pop, &conf)
			assert.NoError(b, err, "failed to run sequential epoch executor")
		}
		log.Printf("sequential %s", time.Since(start))
	})

	b.Run("parallel", func(b *testing.B) {
		var start = time.Now()
		conf := neat.Options{
			CompatThreshold: 0.5,
			DropOffAge:      1,
			PopSize:         200,
			BabiesStolen:    10,
			RecurOnlyProb:   0.2,
		}
		neat.LogLevel = neat.LogLevelInfo
		gen := newGenomeRand(1, in, out, n, nmax, false, linkProb)
		pop, err := NewPopulation(gen, &conf)
		require.NoError(b, err, "failed to create population")
		require.NotNil(b, pop, "population expected")
		for i := 0; i < b.N; i++ {
			err = parallelExecutorNextEpoch(pop, &conf)
			assert.NoError(b, err, "failed to run parallel epoch executor")
		}
		log.Printf("parallel %s", time.Since(start))
	})
}
