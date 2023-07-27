conda activate hdf

cd /tmpdata/dz2449/${SLURM_JOB_ID}
cp -r $SCRATCH/pdf_test .
cd pdf_test

for num_examples in 2 4 8 16 20 32;
then
  python create_datasets.py ${i} 0 0
  mv dense_times.pkl $SCRATCH/pdf_test/dense_times_tmpdata_${i}.pkl
  python main.py
  mv speed_test_datasets.json $SCRATCH/pdf_test/speed_test_datasets_tmpdata_${i}.json
done

cd $SCRATCH/pdf_test
for num_examples in 2 4 8 16 20 32;
then
  python create_datasets.py ${i} 0 0
  mv dense_times.pkl $SCRATCH/pdf_test/dense_times_scratch_${i}.pkl
  python main.py
  mv speed_test_datasets.json $SCRATCH/pdf_test/speed_test_datasets_scratch_${i}.jsoncd
done


