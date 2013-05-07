#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Int_Input.H>
#include <FL/Fl_Slider.H>
#include <stdio.h>

// sliderinput -- simple example of tying an fltk slider and input widget together
// 1.00 erco 10/17/04

class SliderInput : public Fl_Group {
    Fl_Int_Input *input;
    Fl_Slider    *slider;

    // CALLBACK HANDLERS
    //    These 'attach' the input and slider's values together.
    //
    void Slider_CB2() {
        static int recurse = 0;
        if ( recurse ) { 
            return;
        } else {
            recurse = 1;
            char s[80];
            sprintf(s, "%d", (int)(slider->value() + .5));
            // fprintf(stderr, "SPRINTF(%d) -> '%s'\n", (int)(slider->value()+.5), s);
            input->value(s);          // pass slider's value to input
            recurse = 0;
        }
    }

    static void Slider_CB(Fl_Widget *w, void *data) {
        ((SliderInput*)data)->Slider_CB2();
    }

    void Input_CB2() {
        static int recurse = 0;
        if ( recurse ) {
            return;
        } else {
            recurse = 1;
            int val = 0;
            if ( sscanf(input->value(), "%d", &val) != 1 ) {
                val = 0;
            }
            // fprintf(stderr, "SCANF('%s') -> %d\n", input->value(), val);
            slider->value(val);         // pass input's value to slider
            recurse = 0;
        }
    }
    static void Input_CB(Fl_Widget *w, void *data) {
        ((SliderInput*)data)->Input_CB2();
    }

public:
    // CTOR
    SliderInput(int x, int y, int w, int h, const char *l=0) : Fl_Group(x,y,w,h,l) {
        int in_w = 40;
        input  = new Fl_Int_Input(x, y, in_w, h);
        input->callback(Input_CB, (void*)this);
        input->when(FL_WHEN_ENTER_KEY|FL_WHEN_NOT_CHANGED);

        slider = new Fl_Slider(x+in_w, y, w-in_w, h);
        slider->type(1);
        slider->callback(Slider_CB, (void*)this);

        bounds(1, 10);     // some usable default
        value(5);          // some usable default
        end();             // close the group
    }

    // MINIMAL ACCESSORS --  Add your own as needed
    int  value() const    { return((int)(slider->value() + 0.5)); }
    void value(int val)   { slider->value(val); Slider_CB2(); }
    void minumum(int val) { slider->minimum(val); }
    int  minumum() const  { return((int)slider->minimum()); }
    void maximum(int val) { slider->maximum(val); }
    int  maximum() const  { return((int)slider->maximum()); }
    void bounds(int low, int high) { slider->bounds(low, high); }
};