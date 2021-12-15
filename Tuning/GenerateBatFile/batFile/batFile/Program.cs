using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace batFile
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataset = "\"D:/Deep/LPIPS_modified/dataset/2afc/train/TexturedDB_80%%_TrainList_withnbPatchesPerVP_threth0.6_shuffled.csv\"" ;

            float[] lr = {0.0001f, 0.0005f, 0.001f}; // learning rate

            int[] nPatches = { 32, 60, 150};
            int[] nInputImages = { 1, 4, 10};
            //int batch_size = nPatches*nInputImages

            int[] nepoch        = {  1,  3,  5,  7,  5, 12, 20, 15, 25, 35};
            int[] nepoch_decay  = {  9,  7,  5,  3, 20, 13,  5, 35, 25, 15};
            //int[] Total_epoch = { 10, 10, 10, 10, 25, 25, 25, 50, 50, 50}; Total_epoch = nepoch + nepoch_decay

            string output_batFile = "D:/Deep/LPIPS_modified/Lancheur_Tuning.bat";
            StreamWriter sw = new StreamWriter(output_batFile);

            for (int i = 0; i < nepoch.Length; i++)
            {
                int nbepoch = nepoch[i];
                int nbepochDecay = nepoch_decay[i];

                int testset_freq = 5;
                if (nbepoch + nbepochDecay == 10)
                    testset_freq = 2;
                else if (nbepoch + nbepochDecay == 50)
                    testset_freq = 10;

                foreach (int nbpat in nPatches)
                    foreach (int nbinp in nInputImages)
                        foreach (float l in lr)
                        {
                            string s = "python train.py --datasets " + dataset + " --net alex --use_gpu --train_plot --name Tuning --nThreads 10 --npatches " + nbpat + " --nInputImg " + nbinp +
                                " --nepoch " + nbepoch + " --nepoch_decay " + nbepochDecay + " --lr " + l + " --testset_freq " + testset_freq;
                            sw.WriteLine(s);
                        }
            }
            sw.Close();
        }
    }
}
