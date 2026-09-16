import matplotlib.pyplot as plt

def huatu(train_losses, test_losses, test_accuracies, test_aucs):
    # 绘制收敛曲线,epochs为x轴,loss accuracy auc为y轴
    epochs = range(len(train_losses))
    plt.figure(figsize=(6, 3))
    # 绘制训练损失曲线
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    # 绘制测试损失曲线
    plt.plot(epochs, test_losses, label='Test Loss', color='orange')
    # 绘制测试准确率曲线
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    # 绘制测试AUC曲线
    plt.plot(epochs, test_aucs, label='Test AUC', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Convergence Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

