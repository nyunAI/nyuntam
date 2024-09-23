# Contributing to Nyuntam

Thank you for your interest in contributing to Nyuntam! Whether you're fixing a bug, adding a feature, or improving our documentation, your contribution is greatly appreciated. This guide will help you navigate the process of contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Repository Structure](#understanding-the-repository-structure)
3. [How to Contribute](#how-to-contribute)
   - [Reporting Issues](#reporting-issues)
   - [Submitting Pull Requests](#submitting-pull-requests)
   - [Adding New Features](#adding-new-features)
   - [Improving Documentation](#improving-documentation)
4. [Code Style and Guidelines](#code-style-and-guidelines)
5. [Community and Support](#community-and-support)
<!-- 6. [Running Tests](#running-tests) -->

## Getting Started

To get started with contributing to Nyuntam:

1. **Fork the Repository**: Click the "Fork" button on the [Nyuntam GitHub page](https://github.com/nyunAI/nyuntam) to create your copy of the repository.
2. **Clone Your Fork**:

    ```bash
    git clone https://github.com/your-username/nyuntam.git
    cd nyuntam
    ```
   Note: For any contributions across nyuntam and its subsequent submodules (nyuntam-text-generation, nyuntam-vision, or nyuntam-adapt) make sure you do a recursive submodule init to add the submodule(s)

    ```bash
    # based on what part of the code you are working on
    git submodule update --init --recursive <submodule>
    ```
   where submodule can be text_generation, vision or nyuntam_adapt. Post that, change the directory in the submodule you work on and follow these steps,

3. **Install Dependencies**: Depending on which part of the project you're working on, you might need to set up a virtual environment or a Docker container. Refer to the [Installation section](#installation) in the README for detailed instructions.

4. **Set Up Your Branch**: Prefer creating a new branch for your work.

    ```bash
    git checkout -b feature/your-feature-name
    ```

## Understanding the Repository Structure

Nyuntam is organized into several submodules, each serving a specific purpose. Here’s a quick overview:

- **Root Directory**: Contains the core scripts like `main.py`, `algorithm.py`, and `commands.py` that drive the primary functionalities of Nyuntam.
  
- **examples/**: This is where you can find practical examples for different tasks, such as text generation, vision models, and adaptation tasks. Each subdirectory here includes a `README.md` for guidance and `config.yaml` files for configurations.

- **nyuntam_adapt/**, **text_generation/**, **vision/**: These directories contain the main modules for adaptation, text generation, and vision-related tasks. They include the core algorithms, tasks, and utilities, along with relevant `README.md` files and configurations.

- **utils/**: Utility scripts and functions that support the various modules in the repository.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/nyunAI/nyuntam/issues/new/choose) on GitHub. When reporting an issue, be sure to include:

- A clear and descriptive title.
- A detailed description of the problem or suggestion.
- Steps to reproduce the issue (if applicable).
- Any relevant logs, screenshots, or code snippets.

### Submitting Pull Requests

1. **Create a Branch**: Make sure to branch off from the `main` branch.

    ```bash
    git checkout -b your-branch-name
    ```

2. **Make Your Changes**: Work on your feature, bug fix, or documentation update in your branch.

3. **Test Your Changes**: Before submitting, make sure your changes work as expected and do not break existing functionality. See [Running Tests](#running-tests) for guidance.

4. **Commit Your Changes**: Write clear and concise commit messages.

    ```bash
    git commit -m "Description of your changes"
    ```

5. **Push to GitHub**:

    ```bash
    git push origin your-branch-name
    ```

6. **Open a Pull Request**: Navigate to your fork on GitHub and open a pull request against the `main` branch of the original repository. Provide a clear description of what your PR does and any additional context that might be needed for the review.

### Adding New Features

If you’re adding a new feature:

- **Check for Existing Implementations**: Before you start, review the existing modules to see if your feature overlaps with any current functionality.
- **Follow the Repository Structure**: Place your new code in the appropriate module. For example, if your feature involves a new text generation algorithm, it should go under `text_generation/`.
- **Update Documentation**: Ensure that any new features are well-documented, both in the code (via docstrings) and in the relevant `README.md` files.

### Improving Documentation

Documentation is crucial for any open-source project. If you notice any gaps or errors in the documentation:

- **Fix Typos or Errors**: If you spot a typo or an error, feel free to correct it and submit a PR.
- **Expand on Existing Documentation**: If a section of the documentation is unclear or lacking detail, you can improve it by adding more information or examples.

## Code Style and Guidelines

To maintain consistency across the project, please adhere to the following guidelines:

- **PEP 8**: Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- **Code Reviews**: Be open to feedback during the code review process. It’s an opportunity to learn and improve your contributions.

<!-- ## Running Tests

Testing is a critical part of ensuring the reliability of the project. Before submitting your changes, make sure to run the existing tests and add new ones if applicable.

- **Unit Tests**: Check for any existing tests related to your changes. If none exist, consider writing unit tests to cover new functionality.
- **Integration Tests**: For more complex features, integration tests that cover the interaction between different modules are encouraged.
- **Running Tests**: Use the following command to run tests:

    ```bash
    pytest
    ```

- **Docker Tests**: If your changes involve Docker configurations or containers, ensure that your tests run smoothly in those environments. -->

## Community and Support

If you need help or have any questions, feel free to reach out:

- **Contribute to Issues**: Browse the open issues and see where you can help out.
- **Follow Us on Social Media**: Stay updated with the latest news and announcements.

Thank you for contributing to Nyuntam! Together, we can create powerful tools to optimize and adapt deep learning models.
