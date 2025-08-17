/*
  ==============================================================================

	This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginEditor.h"
#include "PluginProcessor.h"

//==============================================================================
_4A2AAudioProcessorEditor::_4A2AAudioProcessorEditor(
	_4A2AAudioProcessor &p, juce::AudioProcessorValueTreeState &vts)
	: AudioProcessorEditor(&p), audioProcessor(p), valueTreeState(vts)
{
	// Make sure that before the constructor has finished, you've set the
	// editor's size to whatever you need it to be.
	setSize(400, 300);

	modeSwitch.setClickingTogglesState(true);
	modeSwitch.setColour(juce::TextButton::buttonOnColourId, juce::Colours::dimgrey);
	modeSwitch.onClick = [this]
	{
		auto mode = modeSwitch.getToggleState();
		modeSwitch.setButtonText(mode ? "Limiter Mode" : "Compressor Mode");
		auto peakValue = peakReduction.getValue();
		auto params = interp(peakValue, mode);
		threshold.setValue(params[0]);
		ratio.setValue(params[1]);
		attackMs.setValue(params[2]);
		releaseMs.setValue(params[3]);
		makeUp.setValue(params[4]);
	};
	modeSwitch.setButtonText("Compressor Mode");

	peakReduction.setRange(40, 100);

	peakReduction.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
								  peakReduction.getTextBoxHeight());
	threshold.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
							  threshold.getTextBoxHeight());
	ratio.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
						  ratio.getTextBoxHeight());
	attackMs.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
							 attackMs.getTextBoxHeight());
	releaseMs.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
							  releaseMs.getTextBoxHeight());
	makeUp.setTextBoxStyle(juce::Slider::TextBoxLeft, false, 120,
						   makeUp.getTextBoxHeight());

	peakReductionLabel.setText("Peak Reduction", juce::dontSendNotification);
	peakReductionLabel.attachToComponent(&peakReduction, true);

	thresholdLabel.setText("Threshold", juce::dontSendNotification);
	thresholdLabel.attachToComponent(&threshold, true);
	threshold.setTextValueSuffix(" dB");

	ratioLabel.setText("Ratio", juce::dontSendNotification);
	ratioLabel.attachToComponent(&ratio, true);

	attackMsLabel.setText("Attack", juce::dontSendNotification);
	attackMsLabel.attachToComponent(&attackMs, true);
	attackMs.setTextValueSuffix(" ms");

	releaseMsLabel.setText("Release", juce::dontSendNotification);
	releaseMsLabel.attachToComponent(&releaseMs, true);
	releaseMs.setTextValueSuffix(" ms");

	makeUpLabel.setText("Make-up", juce::dontSendNotification);
	makeUpLabel.attachToComponent(&makeUp, true);
	makeUp.setTextValueSuffix(" dB");

	peakReduction.onValueChange = [this]
	{
		auto mode = modeSwitch.getToggleState();
		auto params = interp(peakReduction.getValue(), mode);
		threshold.setValue(params[0]);
		ratio.setValue(params[1]);
		attackMs.setValue(params[2]);
		releaseMs.setValue(params[3]);
		makeUp.setValue(params[4]);
	};

	getLookAndFeel().setColour(juce::Slider::thumbColourId, juce::Colours::red);
	getLookAndFeel().setColour(juce::Label::textColourId, juce::Colours::black);

	// Manually set the text box colour as look&feel doesn't seem to work
	peakReduction.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);
	threshold.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);
	ratio.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);
	attackMs.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);
	releaseMs.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);
	makeUp.setColour(juce::Slider::textBoxTextColourId, juce::Colours::black);

	addAndMakeVisible(&peakReduction);
	addAndMakeVisible(&peakReductionLabel);
	addAndMakeVisible(&threshold);
	addAndMakeVisible(&thresholdLabel);
	addAndMakeVisible(&ratio);
	addAndMakeVisible(&ratioLabel);
	addAndMakeVisible(&attackMs);
	addAndMakeVisible(&attackMsLabel);
	addAndMakeVisible(&releaseMs);
	addAndMakeVisible(&releaseMsLabel);
	addAndMakeVisible(&makeUp);
	addAndMakeVisible(&makeUpLabel);
	addAndMakeVisible(&modeSwitch);

	thresholdAttachment.reset(
		new SliderAttachment(valueTreeState, "th", threshold));
	ratioAttachment.reset(new SliderAttachment(valueTreeState, "ratio", ratio));
	attackMsAttachment.reset(
		new SliderAttachment(valueTreeState, "at", attackMs));
	releaseMsAttachment.reset(
		new SliderAttachment(valueTreeState, "rt", releaseMs));
	makeUpAttachment.reset(
		new SliderAttachment(valueTreeState, "makeUp", makeUp));
	modeAttachment.reset(
		new ButtonAttachment(valueTreeState, "mode", modeSwitch));
	peakReductionAttachment.reset(
		new SliderAttachment(valueTreeState, "peak", peakReduction));
}

_4A2AAudioProcessorEditor::~_4A2AAudioProcessorEditor()
{
	thresholdAttachment.reset();
	ratioAttachment.reset();
	attackMsAttachment.reset();
	releaseMsAttachment.reset();
	makeUpAttachment.reset();
	modeAttachment.reset();
	peakReductionAttachment.reset();
}

//==============================================================================
void _4A2AAudioProcessorEditor::paint(juce::Graphics &g)
{
	// fill the whole window white
	g.fillAll(juce::Colours::white);

	// set the current drawing colour to black
	g.setColour(juce::Colours::black);

	// set the font size and draw text to the screen
	g.setFont(25.0f);

	g.drawFittedText("4A-2A", 0, 0, getWidth(), 30,
					 juce::Justification::centred, 1);
}

void _4A2AAudioProcessorEditor::resized()
{
	// This is generally where you'll want to lay out the positions of any
	// subcomponents in your editor..
	auto sliderLeft = 120;
	threshold.setBounds(sliderLeft, 40, getWidth() - sliderLeft - 10, 20);
	ratio.setBounds(sliderLeft, 70, getWidth() - sliderLeft - 10, 20);
	attackMs.setBounds(sliderLeft, 100, getWidth() - sliderLeft - 10, 20);
	releaseMs.setBounds(sliderLeft, 130, getWidth() - sliderLeft - 10, 20);
	makeUp.setBounds(sliderLeft, 160, getWidth() - sliderLeft - 10, 20);

	peakReduction.setBounds(sliderLeft, 220, getWidth() - sliderLeft - 10, 20);
	modeSwitch.setBounds(sliderLeft, 250, getWidth() - sliderLeft - 10, 20);
}

std::array<float, 5> _4A2AAudioProcessorEditor::interp(float peakValue, bool limitMode)
{
	auto *paramPoints = limitMode ? limitPoints : compPoints;
	auto normalisedPeak = (peakValue - peakPoints[0]) * 0.2;
	int lower = std::min(11, (int)normalisedPeak);
	int upper = lower + 1;
	auto p = normalisedPeak - lower;
	std::array<float, 5> interpParam;
	std::transform(std::begin(paramPoints[upper]), std::end(paramPoints[upper]),
				   std::begin(paramPoints[lower]), interpParam.begin(),
				   [&p](auto u, auto l)
				   { return p * u + (1 - p) * l; });
	return interpParam;
}
