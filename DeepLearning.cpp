#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;
using namespace torch::data;

const std::string kDataRoot = R"(..\..\data)";

const int64_t BATCH = 64, EPOCHS = 2, kLogInterval = 10;

struct Net : Module {
    Net() :        
        fc1(784, 1000),
        fc2(1000, 10) 
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    Tensor forward(Tensor x) {
        x = x.view({ -1, 784 });
        x = relu(fc1->forward(x));       
        x = fc2->forward(x);
        return log_softmax(x, 1);
    }

    Linear fc1;
    Linear fc2;
};


auto main() -> int {
    Device device(kCPU);

    Net model;
    model.to(device);

    auto train_dataset = datasets::MNIST(kDataRoot)
        .map(transforms::Normalize<>(0.1307, 0.3081))
        .map(transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
        make_data_loader<samplers::SequentialSampler>(std::move(train_dataset), BATCH);

    optim::SGD optimizer(model.parameters(), optim::SGDOptions(0.01).momentum(0.5));

    model.train();
    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        size_t batch_idx = 0;
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward(data);
            auto loss = nll_loss(output, targets);
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();

            if (batch_idx++ % kLogInterval == 0) {
                std::printf(
                    "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f\n",
                    epoch,
                    batch_idx * batch.data.size(0),
                    train_dataset_size,
                    loss.template item<float>());
            }
        }
     }
}